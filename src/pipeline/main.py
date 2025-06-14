# src/pipeline/main.py
"""
Pipeline v5 – Final. Reemplaza el filtro RL por un clasificador supervisado (LightGBM).

Cambios clave:
• Se eliminan los componentes `prepare_rl_data` y `train_rl`.
• Se añade el nuevo componente `train_filter_model`.
• El backtest final ahora usa el filtro LightGBM en lugar del agente PPO.
• Esta versión es más robusta, eficiente y fácil de validar.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import google.cloud.aiplatform as aip
from kfp import compiler, dsl
from kfp.components import load_component_from_text

from src.shared import constants

# ───────────────────────────── CLI ──────────────────────────────
parser = argparse.ArgumentParser("Compila y/o envía la pipeline v5 final")
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

# Cargar todos los componentes necesarios para la v5
component_op_factory = {
    "data_ingestion":            load_utf8_component("data_ingestion/component.yaml"),
    "data_preparation":          load_utf8_component("data_preparation/component.yaml"),
    "optimize_model_architecture": load_utf8_component("optimize_model_architecture/component.yaml"),
    "optimize_trading_logic":    load_utf8_component("optimize_trading_logic/component.yaml"),
    "train_lstm_launcher":       load_utf8_component("train_lstm_launcher/component.yaml"),
    "train_filter_model":        load_utf8_component("train_filter_model/component.yaml"), # <-- NUEVO
    "backtest":                  load_utf8_component("backtest/component.yaml"),
    "model_promotion":           load_utf8_component("model_promotion/component.yaml"),
}

# Asignar la misma imagen Docker a todos los contenedores
for comp in component_op_factory.values():
    if hasattr(comp.component_spec.implementation, 'container') and comp.component_spec.implementation.container:
        comp.component_spec.implementation.container.image = args.common_image_uri

# ───────────────────────── Definición PIPELINE v5 ─────────────────────────
@dsl.pipeline(
    name="algo-trading-mlops-pipeline-v5-supervised-filter",
    description="Versión final: HPO Secuencial -> LSTM -> Filtro LightGBM -> Backtest -> Promoción",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline_v5(
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials_arch: int = 20,
    n_trials_logic: int = 30,
    backtest_years_to_keep: int = 5,
    holdout_months: int = 3,
):
    # --- PASOS GLOBALES ---

    # 1 ▸ Ingestión y 2 ▸ Preparación de datos (con hold-out)
    ingest_task = component_op_factory["data_ingestion"](timeframe=timeframe)
    prepare_opt_data_task = component_op_factory["data_preparation"](
        timeframe=timeframe,
        years_to_keep=backtest_years_to_keep,
        holdout_months=holdout_months,
    ).after(ingest_task)

    # 3 ▸ Optimizar Arquitectura y 4 ▸ Optimizar Lógica de Trading
    optimize_arch_task = component_op_factory["optimize_model_architecture"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"],
        n_trials=n_trials_arch
    )
    optimize_logic_task = component_op_factory["optimize_trading_logic"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"],
        architecture_params_dir=optimize_arch_task.outputs["best_architecture_dir"],
        n_trials=n_trials_logic
    )
    for task in [optimize_arch_task, optimize_logic_task]:
        task.set_cpu_limit("8").set_memory_limit("30G").set_accelerator_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT).set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE)

    # --- BUCLE PARALELO POR CADA PAR ---
    
    pairs_to_process = list(constants.SPREADS_PIP.keys())
    
    with dsl.ParallelFor(items=pairs_to_process, name="parallel-training-for-each-pair") as pair:
        
        # 5 ▸ Entrenamiento LSTM para el par actual
        train_lstm_task = component_op_factory["train_lstm_launcher"](
            project_id=constants.PROJECT_ID,
            region=constants.REGION,
            pair=pair,
            timeframe=timeframe,
            params_path=f"{optimize_logic_task.outputs['best_params_dir']}/{pair}/best_params.json",
            features_gcs_path=prepare_opt_data_task.outputs["prepared_data_path"],
            output_gcs_base_dir=constants.LSTM_MODELS_PATH,
            vertex_training_image_uri=args.common_image_uri,
            vertex_machine_type=constants.DEFAULT_VERTEX_GPU_MACHINE_TYPE,
            vertex_accelerator_type=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE,
            vertex_accelerator_count=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT,
            vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
        )

        # 6 ▸ Entrenar Modelo de Filtro (LightGBM) para el par actual
        train_filter_task = component_op_factory["train_filter_model"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            features_path=prepare_opt_data_task.outputs["prepared_data_path"],
            pair=pair,
            timeframe=timeframe,
            output_gcs_base_dir=constants.FILTER_MODELS_PATH, # Añade FILTER_MODELS_PATH a constants.py
        )

        # 7 ▸ Backtest Final sobre Datos Hold-Out
        backtest_task = component_op_factory["backtest"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            filter_model_path=train_filter_task.outputs["trained_filter_model_path"], # <-- ¡CAMBIO IMPORTANTE!
            features_path=prepare_opt_data_task.outputs["holdout_data_path"],
            pair=pair,
            timeframe=timeframe,
        )

        # 8 ▸ Promoción a Producción
        component_op_factory["model_promotion"](
            new_metrics_dir=backtest_task.outputs["output_gcs_dir"],
            new_lstm_artifacts_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            new_filter_model_path=train_filter_task.outputs["trained_filter_model_path"], # <-- ¡CAMBIO IMPORTANTE!
            pair=pair,
            timeframe=timeframe,
            production_base_dir=constants.PRODUCTION_MODELS_PATH,
        ).after(backtest_task)


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PIPELINE_JSON = "algo_trading_mlops_pipeline_v5.json"

    compiler.Compiler().compile(trading_pipeline_v5, PIPELINE_JSON)
    print(f"✅ Pipeline v5 compilada a {PIPELINE_JSON}")

    if os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() == "true":
        aip.init(project=constants.PROJECT_ID, location=constants.REGION)
        display_name = f"algo-trading-v5-supervised-filter-{datetime.utcnow():%Y%m%d-%H%M%S}"
        job = aip.PipelineJob(
            display_name=display_name,
            template_path=PIPELINE_JSON,
            pipeline_root=constants.PIPELINE_ROOT,
            enable_caching=False,
            parameter_values={
                "timeframe": constants.DEFAULT_TIMEFRAME,
            },
        )
        print(f"🚀 Enviando PipelineJob '{display_name}'")
        job.run()
        print("📊 Sigue el progreso en Vertex AI → Pipelines")