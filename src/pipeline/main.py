# -----------------------------------------------------------------------------
# main.py: Definición y Ejecución de la Pipeline de MLOps v5
# -----------------------------------------------------------------------------
# Versión final con estructura de bucle corregida y gestión de rutas robusta.
# -----------------------------------------------------------------------------

import argparse
import os
from datetime import datetime
from pathlib import Path

import google.cloud.aiplatform as aip
from kfp import dsl
from kfp.compiler import Compiler
from kfp.components import load_component_from_text

from src.shared import constants

# --- Configuración del CLI ---
parser = argparse.ArgumentParser("Compila y/o envía la pipeline v5")
parser.add_argument(
    "--common-image-uri",
    required=True,
    help="URI Docker – MISMA imagen para todos los componentes",
)
args, _ = parser.parse_known_args()

# --- Carga de Componentes ---
COMPONENTS_DIR = Path(__file__).parent.parent / "components"

def load_utf8_component(rel_path: str):
    """Carga un componente YAML preservando UTF-8 (Windows-safe)."""
    yaml_text = (COMPONENTS_DIR / rel_path).read_text(encoding="utf-8")
    return load_component_from_text(yaml_text)

component_op_factory = {
    "data_ingestion": load_utf8_component("data_ingestion/component.yaml"),
    "data_preparation": load_utf8_component("data_preparation/component.yaml"),
    "optimize_model_architecture": load_utf8_component("optimize_model_architecture/component.yaml"),
    "optimize_trading_logic": load_utf8_component("optimize_trading_logic/component.yaml"),
    "train_lstm_launcher": load_utf8_component("train_lstm_launcher/component.yaml"),
    "train_filter_model": load_utf8_component("train_filter_model/component.yaml"),
    "backtest": load_utf8_component("backtest/component.yaml"),
    "model_promotion": load_utf8_component("model_promotion/component.yaml"),
}

for comp in component_op_factory.values():
    if hasattr(comp.component_spec.implementation, "container"):
        comp.component_spec.implementation.container.image = args.common_image_uri


# --- Definición de la Pipeline Corregida ---
@dsl.pipeline(
    name="algo-trading-mlops-pipeline-v5-robust-paths",
    description="Versión final con estructura de bucle corregida.",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline_v5(
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials_arch: int = 20,
    n_trials_logic: int = 30,
    backtest_years_to_keep: int = 5,
    holdout_months: int = 3,
):
    """Pipeline de entrenamiento y despliegue para IA de trading Forex."""
    
    pairs_to_process = list(constants.SPREADS_PIP.keys())

    # AJUSTE ESTRUCTURAL: El bucle ahora engloba las tareas que dependen del par.
    with dsl.ParallelFor(
        items=pairs_to_process, name="parallel-processing-for-each-pair"
    ) as pair:

        # 1 ▸ Ingestión de datos (AHORA DENTRO DEL BUCLE)
        # Se ejecuta una vez por cada par, pasándole el par correcto.
        ingest_task = component_op_factory["data_ingestion"](
            project_id=constants.PROJECT_ID,
            polygon_secret_name=constants.POLYGON_API_KEY_SECRET_NAME,
            end_date=datetime.utcnow().strftime("%Y-%m-%d"),
            timeframe=timeframe,
            pair=pair
        )
        ingest_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        # 2 ▸ Preparación de datos (AHORA DENTRO DEL BUCLE)
        # Recibe el par del bucle y ahora sí buscará el archivo correcto.
        prepare_data_task = component_op_factory["data_preparation"](
            pair=pair,
            timeframe=timeframe,
            years_to_keep=backtest_years_to_keep,
            holdout_months=holdout_months,
        ).after(ingest_task)
        prepare_data_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
        
        # 3 ▸ Optimizar arquitectura LSTM
        # Usa los datos de entrenamiento específicos del par.
        optimize_arch_task = component_op_factory["optimize_model_architecture"](
            features_path=prepare_data_task.outputs["prepared_data_path"],
            n_trials=n_trials_arch,
            pair=pair
        )
        optimize_arch_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        # 4 ▸ Optimizar lógica de trading
        optimize_logic_task = component_op_factory["optimize_trading_logic"](
            features_path=prepare_data_task.outputs["prepared_data_path"],
            architecture_params_file=f"{optimize_arch_task.outputs['best_architecture_dir']}/{pair}/best_architecture.json",
            n_trials=n_trials_logic,
        )
        optimize_logic_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        # 5 ▸ Entrenamiento LSTM
        train_lstm_task = component_op_factory["train_lstm_launcher"](
            project_id=constants.PROJECT_ID,
            region=constants.REGION,
            pair=pair,
            timeframe=timeframe,
            params_file=f"{optimize_logic_task.outputs['best_params_dir']}/{pair}/best_params.json",
            features_gcs_path=prepare_data_task.outputs["prepared_data_path"],
            output_gcs_base_dir=constants.LSTM_MODELS_PATH,
            vertex_training_image_uri=args.common_image_uri,
            vertex_machine_type=constants.DEFAULT_VERTEX_GPU_MACHINE_TYPE,
            vertex_accelerator_type=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE,
            vertex_accelerator_count=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT,
            vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
        )
        train_lstm_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        # 6 ▸ Entrenamiento del modelo filtro
        train_filter_task = component_op_factory["train_filter_model"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            features_path=prepare_data_task.outputs["prepared_data_path"],
            pair=pair,
            timeframe=timeframe,
            output_gcs_base_dir=constants.FILTER_MODELS_PATH,
        )
        train_filter_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        # 7 ▸ Backtest final
        backtest_task = component_op_factory["backtest"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            filter_model_path=train_filter_task.outputs["trained_filter_model_path"],
            features_path=prepare_data_task.outputs["holdout_data_path"],
            pair=pair,
            timeframe=timeframe,
        )
        backtest_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        # 8 ▸ Promoción a producción
        promotion_task = component_op_factory["model_promotion"](
            new_metrics_dir=backtest_task.outputs["output_gcs_dir"],
            new_lstm_artifacts_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            new_filter_model_path=train_filter_task.outputs["trained_filter_model_path"],
            pair=pair,
            timeframe=timeframe,
            production_base_dir=constants.PRODUCTION_MODELS_PATH,
        )
        promotion_task.after(backtest_task)
        promotion_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)


# --- Bloque de Ejecución ---
if __name__ == "__main__":
    PIPELINE_JSON = "algo_trading_mlops_pipeline_v5_corrected.json"
    Compiler().compile(trading_pipeline_v5, PIPELINE_JSON)
    print(f"✅ Pipeline v5 (estructura corregida) compilada a {PIPELINE_JSON}")

    if os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() == "true":
        aip.init(project=constants.PROJECT_ID, location=constants.REGION)
        display_name = f"algo-trading-v5-final-{datetime.utcnow():%Y%m%d-%H%M%S}"
        job = aip.PipelineJob(
            display_name=display_name,
            template_path=PIPELINE_JSON,
            pipeline_root=constants.PIPELINE_ROOT,
            enable_caching=True,
        )
        job.run(service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT)
        print(f"🚀 Pipeline lanzada con Display Name: {display_name}")
    else:
        print("⏭️ La pipeline no se envió a Vertex AI.")