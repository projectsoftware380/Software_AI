# -----------------------------------------------------------------------------
# training_only_pipeline.py: Pipeline de solo entrenamiento para GCP.
# -----------------------------------------------------------------------------
#
# Esta pipeline está diseñada para ejecutarse en Vertex AI y se centra
# exclusivamente en las tareas de uso intensivo de cómputo:
# - Optimización de arquitectura.
# - Optimización de lógica de trading.
# - Entrenamiento de modelos (LSTM y filtro).
#
# Asume que los datos ya han sido preparados y están disponibles en GCS.
# -----------------------------------------------------------------------------

import logging
from pathlib import Path
from kfp import dsl

from src.shared import constants
from src.utils.kfp_utils import load_component_from_text_utf8
from src.shared.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# --- Carga de Componentes ---
COMPONENTS_DIR = Path(__file__).parent.parent / "components"

def load_component_factory():
    """Carga y configura los componentes de la pipeline."""
    return {
        "optimize_model_architecture": load_component_from_text_utf8(COMPONENTS_DIR / "optimize_model_architecture/component.yaml"),
        "optimize_trading_logic": load_component_from_text_utf8(COMPONENTS_DIR / "optimize_trading_logic/component.yaml"),
        "train_lstm_launcher": load_component_from_text_utf8(COMPONENTS_DIR / "train_lstm_launcher/component.yaml"),
        "train_filter_model": load_component_from_text_utf8(COMPONENTS_DIR / "train_filter_model/component.yaml"),
    }

# --- Definición de la Pipeline de Entrenamiento ---
@dsl.pipeline(
    name="algo-trading-training-only-pipeline",
    description="Pipeline que ejecuta solo los pasos de entrenamiento en Vertex AI.",
    pipeline_root=constants.PIPELINE_ROOT,
)
def training_only_pipeline(
    prepared_data_path: str, # Ruta GCS a los datos ya preparados
    pair: str,
    timeframe: str,
    n_trials_arch: int,
    n_trials_logic: int,
    common_image_uri: str,
):
    """Pipeline de solo entrenamiento para la IA de trading."""

    component_op_factory = load_component_factory()

    optimize_arch_task = component_op_factory["optimize_model_architecture"](
        features_path=prepared_data_path,
        n_trials=n_trials_arch,
        pair=pair
    )
    optimize_arch_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

    optimize_logic_task = component_op_factory["optimize_trading_logic"](
        features_path=prepared_data_path,
        architecture_params_file=f"{optimize_arch_task.outputs['best_architecture_dir']}/best_architecture.json",
        n_trials=n_trials_logic,
        pair=pair
    )
    optimize_logic_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

    train_lstm_task = component_op_factory["train_lstm_launcher"](
        project_id=constants.PROJECT_ID,
        region=constants.REGION,
        pair=pair,
        timeframe=timeframe,
        params_file=f"{optimize_logic_task.outputs['best_params_dir']}/best_params.json",
        features_gcs_path=prepared_data_path,
        output_gcs_base_dir=constants.LSTM_MODELS_PATH,
        vertex_training_image_uri=common_image_uri,
        vertex_machine_type=constants.DEFAULT_VERTEX_GPU_MACHINE_TYPE,
        vertex_accelerator_type=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE,
        vertex_accelerator_count=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT,
        vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
    )
    train_lstm_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

    train_filter_task = component_op_factory["train_filter_model"](
        lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        features_path=prepared_data_path,
        pair=pair,
        timeframe=timeframe,
        output_gcs_base_dir=constants.FILTER_MODELS_PATH,
    )
    train_filter_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
