# -----------------------------------------------------------------------------
# pipeline_definition.py: Definición de la Pipeline de MLOps v5 (Corregido)
# -----------------------------------------------------------------------------
# Versión final con estructura de bucle corregida y gestión de rutas robusta.
# -----------------------------------------------------------------------------

import logging
from pathlib import Path
from kfp import dsl

from src.shared import constants

from src.utils.kfp_utils import load_component_from_text_utf8 # Nuevo

# --- Configuración del Logging ---
from src.shared.logging_config import setup_logging

setup_logging() # Asegurar que el logging esté configurado
logger = logging.getLogger(__name__)

# --- Carga de Componentes ---
COMPONENTS_DIR = Path(__file__).parent.parent / "components"

def load_component_factory():
    """Carga y configura los componentes de la pipeline."""
    component_op_factory = {
        "data_ingestion": load_component_from_text_utf8(COMPONENTS_DIR / "dukascopy_ingestion/component.yaml"), # Cambiado a dukascopy_ingestion
        "data_preparation": load_component_from_text_utf8(COMPONENTS_DIR / "data_preparation/component.yaml"),
        "optimize_model_architecture": load_component_from_text_utf8(COMPONENTS_DIR / "optimize_model_architecture/component.yaml"),
        "optimize_trading_logic": load_component_from_text_utf8(COMPONENTS_DIR / "optimize_trading_logic/component.yaml"),
        "train_lstm_launcher": load_component_from_text_utf8(COMPONENTS_DIR / "train_lstm_launcher/component.yaml"),
        "train_filter_model": load_component_from_text_utf8(COMPONENTS_DIR / "train_filter_model/component.yaml"),
        "backtest": load_component_from_text_utf8(COMPONENTS_DIR / "backtest/component.yaml"),
        "model_promotion": load_component_from_text_utf8(COMPONENTS_DIR / "model_promotion/component.yaml"),
    }

    return component_op_factory


# --- Definición de la Pipeline Corregida ---
@dsl.pipeline(
    name="algo-trading-mlops-pipeline-v5-robust-paths",
    description="Versión final con gestión de rutas centralizada, versionada y robusta.",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline_v5(
    timeframe: str,
    n_trials_arch: int,
    n_trials_logic: int,
    backtest_years_to_keep: int,
    holdout_months: int,
    end_date: str,
    common_image_uri: str,
):
    """Pipeline de entrenamiento y despliegue para IA de trading Forex."""

    component_op_factory = load_component_factory()
    
    pairs_to_process = list(constants.SPREADS_PIP.keys())

    with dsl.ParallelFor(
        items=pairs_to_process, name="parallel-processing-for-each-pair"
    ) as pair:

        ingest_task = component_op_factory["data_ingestion"](
            project_id=constants.PROJECT_ID,
            end_date=end_date, # Usar config
            timeframe=timeframe, # Usar config
            pair=pair
        )
        ingest_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        prepare_data_task = component_op_factory["data_preparation"](
            input_data_path=ingest_task.outputs["output_gcs_path"],
            years_to_keep=backtest_years_to_keep, # Usar config
            holdout_months=holdout_months, # Usar config
        ).after(ingest_task)
        prepare_data_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
        
        optimize_arch_task = component_op_factory["optimize_model_architecture"](
            features_path=prepare_data_task.outputs["prepared_data_path"],
            n_trials=n_trials_arch, # Usar config
            pair=pair
        )
        optimize_arch_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
        
        # --- CORRECCIÓN: Se elimina el {pair} extra de la ruta del archivo de parámetros. ---
        optimize_logic_task = component_op_factory["optimize_trading_logic"](
            features_path=prepare_data_task.outputs["prepared_data_path"],
            architecture_params_file=f"{optimize_arch_task.outputs['best_architecture_dir']}/best_architecture.json",
            n_trials=n_trials_logic, # Usar config
            pair=pair
        )
        optimize_logic_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        train_lstm_task = component_op_factory["train_lstm_launcher"](
            project_id=constants.PROJECT_ID,
            region=constants.REGION,
            pair=pair,
            timeframe=timeframe, # Usar config
            params_file=f"{optimize_logic_task.outputs['best_params_dir']}/best_params.json",
            features_gcs_path=prepare_data_task.outputs["prepared_data_path"],
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
            features_path=prepare_data_task.outputs["prepared_data_path"],
            pair=pair,
            timeframe=timeframe, # Usar config
            output_gcs_base_dir=constants.FILTER_MODELS_PATH,
        )
        train_filter_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        backtest_task = component_op_factory["backtest"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            filter_model_path=train_filter_task.outputs["trained_filter_model_path"],
            features_path=prepare_data_task.outputs["holdout_data_path"],
            pair=pair,
            timeframe=timeframe, # Usar config
        )
        backtest_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)

        promotion_task = component_op_factory["model_promotion"](
            new_metrics_dir=backtest_task.outputs["output_gcs_dir"],
            new_lstm_artifacts_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            new_filter_model_path=train_filter_task.outputs["trained_filter_model_path"],
            pair=pair,
            timeframe=timeframe, # Usar config
            production_base_dir=constants.PRODUCTION_MODELS_PATH,
        )
        promotion_task.after(backtest_task)
        promotion_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
