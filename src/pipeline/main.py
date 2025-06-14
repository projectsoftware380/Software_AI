# src/pipeline/main.py
"""
Pipeline v4 – Refactorizada para optimización secuencial y procesamiento multi-par.

Cambios clave:
• Se divide el HPO en dos fases: `optimize_model_architecture` y `optimize_trading_logic`.
• Se eliminan los parámetros a nivel de pipeline que ahora se manejan internamente
  (como `pair`).
• Se utiliza `dsl.ParallelFor` para ejecutar los pasos de entrenamiento, backtesting
  y promoción en paralelo para cada par de divisas.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

import google.cloud.aiplatform as aip
from kfp import compiler, dsl
from kfp.components import load_component_from_text

from src.shared import constants

# ───────────────────────────── CLI ──────────────────────────────
parser = argparse.ArgumentParser("Compila y/o envía la pipeline v4 refactorizada")
parser.add_argument(
    "--common-image-uri",
    required=True,
    help="URI Docker (misma imagen) para TODOS los componentes",
)
args, _ = parser.parse_known_args()

# ─────────────────────── Infra y utilidades ─────────────────────
COMPONENTS_DIR = Path(__file__).parent.parent / "components"

def load_utf8_component(rel_path: str):
    """Carga un componente YAML preservando UTF-8 (Windows safe)."""
    yaml_text = (COMPONENTS_DIR / rel_path).read_text(encoding="utf-8")
    return load_component_from_text(yaml_text)

# Cargar todos los componentes, incluyendo los nuevos y renombrados
component_op_factory = {
    "data_ingestion":            load_utf8_component("data_ingestion/component.yaml"),
    "data_preparation":          load_utf8_component("data_preparation/component.yaml"),
    "optimize_model_architecture": load_utf8_component("optimize_model_architecture/component.yaml"),
    "optimize_trading_logic":    load_utf8_component("optimize_trading_logic/component.yaml"),
    "train_lstm_launcher":       load_utf8_component("train_lstm_launcher/component.yaml"),
    "prepare_rl_data":           load_utf8_component("prepare_rl_data/component.yaml"),
    "train_rl":                  load_utf8_component("train_rl/component.yaml"),
    "backtest":                  load_utf8_component("backtest/component.yaml"),
    "model_promotion":           load_utf8_component("model_promotion/component.yaml"),
}

# Asignar la misma imagen Docker a todos los contenedores
for comp in component_op_factory.values():
    if hasattr(comp.component_spec.implementation, 'container') and comp.component_spec.implementation.container:
        comp.component_spec.implementation.container.image = args.common_image_uri

# ───────────────────────── Definición PIPELINE v4 ─────────────────────────
@dsl.pipeline(
    name="algo-trading-mlops-pipeline-v4",
    description="Refactorizada: Ingestión -> Prep -> HPO Arch -> HPO Logic -> (Loop: Train -> RL -> Backtest -> Promote)",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline_v4(
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials_arch: int = 20,
    n_trials_logic: int = 30,
    backtest_years_to_keep: int = 5,
):
    # --- PASOS GLOBALES (Se ejecutan una vez para todos los pares) ---

    # 1 ▸ Ingestión de datos para todos los pares
    ingest_task = component_op_factory["data_ingestion"](
        timeframe=timeframe,
        project_id=constants.PROJECT_ID,
        polygon_secret_name=constants.POLYGON_API_KEY_SECRET_NAME,
        start_date="2010-01-01",
        end_date=datetime.utcnow().strftime("%Y-%m-%d"),
    )

    # 2 ▸ Preparar datos para HPO para todos los pares
    prepare_opt_data_task = component_op_factory["data_preparation"](
        timeframe=timeframe,
        years_to_keep=backtest_years_to_keep,
    ).after(ingest_task)

    # 3 ▸ Optimizar Arquitectura del Modelo para todos los pares (GPU)
    optimize_arch_task = component_op_factory["optimize_model_architecture"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"],
        n_trials=n_trials_arch,
    )
    optimize_arch_task.set_cpu_limit("8").set_memory_limit("30G")
    optimize_arch_task.set_accelerator_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT)
    optimize_arch_task.set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE)

    # 4 ▸ Optimizar Lógica de Trading para todos los pares (GPU)
    optimize_logic_task = component_op_factory["optimize_trading_logic"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"],
        architecture_params_dir=optimize_arch_task.outputs["best_architecture_dir"],
        n_trials=n_trials_logic,
    )
    optimize_logic_task.set_cpu_limit("8").set_memory_limit("30G")
    optimize_logic_task.set_accelerator_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT)
    optimize_logic_task.set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE)

    # --- BUCLE PARALELO (Se ejecuta una vez por cada par) ---
    
    # Obtenemos la lista de pares del archivo de constantes para iterar sobre ella
    pairs_to_process = list(constants.SPREADS_PIP.keys())
    
    with dsl.ParallelFor(items=pairs_to_process, name="parallel-training-for-each-pair") as pair:
        
        # 5 ▸ Entrenamiento LSTM para el par actual
        # Construimos las rutas a los parámetros específicos de este par
        params_path = f"{optimize_logic_task.outputs['best_params_dir']}/{pair}/best_params.json"
        features_path_unseen = f"{constants.DATA_PATH}/{pair}/{timeframe}/{pair}_{timeframe}_unseen.parquet"

        train_lstm_task = component_op_factory["train_lstm_launcher"](
            vertex_training_image_uri=args.common_image_uri,
            project_id=constants.PROJECT_ID,
            region=constants.REGION,
            pair=pair,
            timeframe=timeframe,
            params_path=params_path,
            features_gcs_path=prepare_opt_data_task.outputs["prepared_data_path"],
            output_gcs_base_dir=constants.LSTM_MODELS_PATH,
            vertex_machine_type=constants.DEFAULT_VERTEX_GPU_MACHINE_TYPE,
            vertex_accelerator_type=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE,
            vertex_accelerator_count=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT,
            vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
        )

        # 6 ▸ Datos para RL para el par actual
        prepare_rl_data_task = component_op_factory["prepare_rl_data"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            pair=pair,
            timeframe=timeframe,
            output_gcs_base_dir=constants.RL_DATA_INPUTS_PATH,
        )

        # 7 ▸ Entrenar agente RL para el par actual (GPU)
        train_rl_task = component_op_factory["train_rl"](
            params_path=f"{train_lstm_task.outputs['trained_lstm_dir_path']}/params.json",
            rl_data_path=prepare_rl_data_task.outputs["rl_data_path"],
            pair=pair,
            timeframe=timeframe,
            output_gcs_base_dir=constants.RL_MODELS_PATH,
            tensorboard_logs_base_dir=constants.TENSORBOARD_LOGS_PATH,
        )
        train_rl_task.set_cpu_limit("8").set_memory_limit("20G")
        train_rl_task.set_accelerator_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT)
        train_rl_task.set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE)

        # 8 ▸ Backtest para el par actual
        backtest_task = component_op_factory["backtest"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
            features_path=features_path_unseen,
            pair=pair,
            timeframe=timeframe,
        )

        # 9 ▸ Promoción a Producción para el par actual
        component_op_factory["model_promotion"](
            new_metrics_dir=backtest_task.outputs["output_gcs_dir"],
            new_lstm_artifacts_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            new_rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
            pair=pair,
            timeframe=timeframe,
            production_base_dir=constants.PRODUCTION_MODELS_PATH,
        )

# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PIPELINE_JSON = "algo_trading_mlops_pipeline_v4.json"

    compiler.Compiler().compile(trading_pipeline_v4, PIPELINE_JSON)
    print(f"✅ Pipeline v4 compilada a {PIPELINE_JSON}")

    if os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() == "true":
        aip.init(project=constants.PROJECT_ID, location=constants.REGION)
        display_name = f"algo-trading-v4-multi-pair-{datetime.utcnow():%Y%m%d-%H%M%S}"
        job = aip.PipelineJob(
            display_name=display_name,
            template_path=PIPELINE_JSON,
            pipeline_root=constants.PIPELINE_ROOT,
            enable_caching=False,
            parameter_values={
                "timeframe": constants.DEFAULT_TIMEFRAME,
                "n_trials_arch": 20,
                "n_trials_logic": 30,
            },
        )
        print(f"🚀 Enviando PipelineJob '{display_name}'")
        job.run()
        print("📊 Sigue el progreso en Vertex AI → Pipelines")