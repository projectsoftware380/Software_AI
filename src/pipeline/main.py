# src/pipeline/main.py
"""
Pipeline v4.1 – Ajuste final para usar el conjunto de datos Hold-Out.

Cambios clave:
• El componente `data_preparation` ahora produce un conjunto de datos hold-out.
• El `backtest` final ahora utiliza exclusivamente este conjunto de datos hold-out,
  asegurando una evaluación verdaderamente fuera de muestra.
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
parser = argparse.ArgumentParser("Compila y/o envía la pipeline v4.1 refactorizada")
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

# Cargar todos los componentes
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

# ───────────────────────── Definición PIPELINE v4.1 ──────────────────────
@dsl.pipeline(
    name="algo-trading-mlops-pipeline-v4-1-with-holdout",
    description="Refactorizada con Hold-Out: Ingestión -> Prep -> HPO Arch -> HPO Logic -> (Loop: Train -> RL -> Backtest -> Promote)",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline_v4_1(
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials_arch: int = 20,
    n_trials_logic: int = 30,
    backtest_years_to_keep: int = 5,
    holdout_months: int = 3,
):
    # --- PASOS GLOBALES ---

    # 1 ▸ Ingestión de datos para todos los pares
    ingest_task = component_op_factory["data_ingestion"](
        timeframe=timeframe,
        project_id=constants.PROJECT_ID,
        polygon_secret_name=constants.POLYGON_API_KEY_SECRET_NAME,
        start_date="2010-01-01",
        end_date=datetime.utcnow().strftime("%Y-%m-%d"),
    )

    # 2 ▸ Preparar datos para HPO y Hold-Out para todos los pares
    prepare_opt_data_task = component_op_factory["data_preparation"](
        timeframe=timeframe,
        years_to_keep=backtest_years_to_keep,
        holdout_months=holdout_months,
    ).after(ingest_task)

    # 3 ▸ Optimizar Arquitectura del Modelo
    optimize_arch_task = component_op_factory["optimize_model_architecture"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"], # Usa datos de entrenamiento/opt
        n_trials=n_trials_arch,
    )
    optimize_arch_task.set_cpu_limit("8").set_memory_limit("30G")
    optimize_arch_task.set_accelerator_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT)
    optimize_arch_task.set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE)

    # 4 ▸ Optimizar Lógica de Trading
    optimize_logic_task = component_op_factory["optimize_trading_logic"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"], # Usa datos de entrenamiento/opt
        architecture_params_dir=optimize_arch_task.outputs["best_architecture_dir"],
        n_trials=n_trials_logic,
    )
    optimize_logic_task.set_cpu_limit("8").set_memory_limit("30G")
    optimize_logic_task.set_accelerator_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT)
    optimize_logic_task.set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE)

    # --- BUCLE PARALELO POR CADA PAR ---
    
    pairs_to_process = list(constants.SPREADS_PIP.keys())
    
    with dsl.ParallelFor(items=pairs_to_process, name="parallel-training-for-each-pair") as pair:
        
        # 5 ▸ Entrenamiento LSTM
        train_lstm_task = component_op_factory["train_lstm_launcher"](
            # ... argumentos ...
            pair=pair,
            timeframe=timeframe,
            params_path=f"{optimize_logic_task.outputs['best_params_dir']}/{pair}/best_params.json",
            features_gcs_path=prepare_opt_data_task.outputs["prepared_data_path"], # Usa datos de entrenamiento/opt
            # ... más argumentos ...
        )

        # 6 ▸ Datos para RL
        prepare_rl_data_task = component_op_factory["prepare_rl_data"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            pair=pair,
            timeframe=timeframe,
            output_gcs_base_dir=constants.RL_DATA_INPUTS_PATH,
        )

        # 7 ▸ Entrenar agente RL
        train_rl_task = component_op_factory["train_rl"](
            # ... argumentos ...
            pair=pair,
            timeframe=timeframe,
            # ... más argumentos ...
        )

        # 8 ▸ Backtest Final sobre Datos Hold-Out
        # === AJUSTE CLAVE ===
        # El backtest ahora usa el conjunto de datos hold-out, que nunca ha sido
        # visto por el modelo durante el entrenamiento o la optimización.
        backtest_task = component_op_factory["backtest"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
            features_path=prepare_opt_data_task.outputs["holdout_data_path"], # <-- ¡CAMBIO IMPORTANTE!
            pair=pair,
            timeframe=timeframe,
        )

        # 9 ▸ Promoción a Producción
        component_op_factory["model_promotion"](
            # ... argumentos ...
            pair=pair,
            timeframe=timeframe,
            # ... más argumentos ...
        ).after(backtest_task)


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PIPELINE_JSON = "algo_trading_mlops_pipeline_v4_1.json"

    compiler.Compiler().compile(trading_pipeline_v4_1, PIPELINE_JSON)
    print(f"✅ Pipeline v4.1 compilada a {PIPELINE_JSON}")

    if os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() == "true":
        aip.init(project=constants.PROJECT_ID, location=constants.REGION)
        display_name = f"algo-trading-v4-1-{datetime.utcnow():%Y%m%d-%H%M%S}"
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