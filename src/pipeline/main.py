# -----------------------------------------------------------------------------
# main.py: definición y ejecución de la pipeline MLOps v5
# -----------------------------------------------------------------------------

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import google.cloud.aiplatform as aip
from kfp import dsl
from kfp.compiler import Compiler
from kfp.components import load_component_from_text

from src.shared import constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    "Compila y/o envía la pipeline v5 con filtro supervisado"
)
parser.add_argument(
    "--common-image-uri",
    required=True,
    help="URI Docker usada por todos los componentes",
)
args, _ = parser.parse_known_args()

COMPONENTS_DIR = Path(__file__).parent.parent / "components"


def load_utf8_component(rel_path: str):
    """Carga un componente YAML preservando UTF-8."""
    yaml_text = (COMPONENTS_DIR / rel_path).read_text(encoding="utf-8")
    return load_component_from_text(yaml_text)


logger.info("Cargando definiciones de componentes...")
component_op_factory = {
    "data_ingestion": load_utf8_component("data_ingestion/component.yaml"),
    "data_preparation": load_utf8_component("data_preparation/component.yaml"),
    "optimize_model_architecture": load_utf8_component(
        "optimize_model_architecture/component.yaml"
    ),
    "optimize_trading_logic": load_utf8_component(
        "optimize_trading_logic/component.yaml"
    ),
    "train_lstm_launcher": load_utf8_component("train_lstm_launcher/component.yaml"),
    "train_filter_model": load_utf8_component("train_filter_model/component.yaml"),
    "backtest": load_utf8_component("backtest/component.yaml"),
    "model_promotion": load_utf8_component("model_promotion/component.yaml"),
}

logger.info("Asignando imagen Docker común: %s", args.common_image_uri)
for component in component_op_factory.values():
    if hasattr(component.component_spec.implementation, "container"):
        component.component_spec.implementation.container.image = args.common_image_uri


@dsl.pipeline(
    name="algo-trading-mlops-pipeline-v5-robust-paths",
    description="Pipeline modular con rutas y configuración centralizadas.",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline_v5(
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials_arch: int = 20,
    n_trials_logic: int = 30,
    backtest_years_to_keep: int = 5,
    holdout_months: int = 3,
):
    """Pipeline experimental de entrenamiento y evaluación para series temporales."""

    pairs_to_process = list(constants.SPREADS_PIP.keys())

    with dsl.ParallelFor(
        items=pairs_to_process, name="parallel-processing-for-each-pair"
    ) as pair:
        ingest_task = component_op_factory["data_ingestion"](
            project_id=constants.PROJECT_ID,
            polygon_secret_name=constants.POLYGON_API_KEY_SECRET_NAME,
            end_date=datetime.utcnow().strftime("%Y-%m-%d"),
            timeframe=timeframe,
            pair=pair,
        )
        ingest_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        prepare_data_task = component_op_factory["data_preparation"](
            input_data_path=ingest_task.outputs["output_gcs_path"],
            years_to_keep=backtest_years_to_keep,
            holdout_months=holdout_months,
        ).after(ingest_task)
        prepare_data_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        optimize_arch_task = component_op_factory["optimize_model_architecture"](
            features_path=prepare_data_task.outputs["prepared_data_path"],
            n_trials=n_trials_arch,
            pair=pair,
        )
        optimize_arch_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        optimize_logic_task = component_op_factory["optimize_trading_logic"](
            features_path=prepare_data_task.outputs["prepared_data_path"],
            architecture_params_file=(
                f"{optimize_arch_task.outputs['best_architecture_dir']}"
                "/best_architecture.json"
            ),
            n_trials=n_trials_logic,
            pair=pair,
        )
        optimize_logic_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        train_lstm_task = component_op_factory["train_lstm_launcher"](
            project_id=constants.PROJECT_ID,
            region=constants.REGION,
            pair=pair,
            timeframe=timeframe,
            params_file=(
                f"{optimize_logic_task.outputs['best_params_dir']}/best_params.json"
            ),
            features_gcs_path=prepare_data_task.outputs["prepared_data_path"],
            output_gcs_base_dir=constants.LSTM_MODELS_PATH,
            vertex_training_image_uri=args.common_image_uri,
            vertex_machine_type=constants.DEFAULT_VERTEX_GPU_MACHINE_TYPE,
            vertex_accelerator_type=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE,
            vertex_accelerator_count=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT,
            vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
        )
        train_lstm_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        train_filter_task = component_op_factory["train_filter_model"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            features_path=prepare_data_task.outputs["prepared_data_path"],
            pair=pair,
            timeframe=timeframe,
            output_gcs_base_dir=constants.FILTER_MODELS_PATH,
        )
        train_filter_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        backtest_task = component_op_factory["backtest"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            filter_model_path=train_filter_task.outputs["trained_filter_model_path"],
            features_path=prepare_data_task.outputs["holdout_data_path"],
            pair=pair,
            timeframe=timeframe,
        )
        backtest_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

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


if __name__ == "__main__":
    pipeline_json = "algo_trading_mlops_pipeline_v5_final.json"

    logger.info("Compilando pipeline en '%s'...", pipeline_json)
    Compiler().compile(trading_pipeline_v5, pipeline_json)
    logger.info("Pipeline compilada exitosamente.")

    # Por seguridad, compilar no implica ejecutar recursos cloud. El envío a
    # Vertex AI es opt-in mediante SUBMIT_PIPELINE_TO_VERTEX=true.
    submit_to_vertex = os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "false").lower() == "true"

    if submit_to_vertex:
        try:
            logger.info("Enviando pipeline a Vertex AI...")
            aip.init(project=constants.PROJECT_ID, location=constants.REGION)

            display_name = f"algo-trading-v5-final-{datetime.utcnow():%Y%m%d-%H%M%S}"
            job = aip.PipelineJob(
                display_name=display_name,
                template_path=pipeline_json,
                pipeline_root=constants.PIPELINE_ROOT,
                enable_caching=True,
            )

            logger.info("Pipeline root: %s", constants.PIPELINE_ROOT)
            logger.info("Service account: %s", constants.VERTEX_LSTM_SERVICE_ACCOUNT)

            job.run(service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT)
            logger.info("Pipeline lanzada: %s", display_name)
        except Exception as exc:
            logger.critical(
                "Fallo al lanzar la pipeline: %s", exc, exc_info=True
            )
            raise
    else:
        logger.info(
            "Pipeline compilada localmente. Para enviarla a Vertex AI define "
            "SUBMIT_PIPELINE_TO_VERTEX=true."
        )
