# ----------------------------------------------------------------
# run_gcp_training.py: Lanza la pipeline de solo entrenamiento.
# ----------------------------------------------------------------

import os
import sys
import logging
from kfp.compiler import Compiler

from src.pipeline.training_only_pipeline import training_only_pipeline
from src.deploy.vertex_ai_runner import run_vertex_pipeline
from src.shared import constants
from src.shared.logging_config import setup_logging

# --- Configuración ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Parámetros de la Pipeline ---
PIPELINE_PARAMETERS = {
    "pair": "EURUSD",
    "timeframe": "m1",
    "n_trials_arch": 15,
    "n_trials_logic": 25,
    "common_image_uri": constants.COMMON_IMAGE_URI,
    "prepared_data_path": f"gs://{constants.PROJECT_ID}-vertex-ai-pipeline-assets/data/prepared/EURUSD/m1/train_opt/prepared_data.parquet"
}

# --- Compilación y Lanzamiento ---
def main():
    """Compila y lanza la pipeline de solo entrenamiento de forma síncrona."""
    pipeline_json_path = "training_only_pipeline.json"
    try:
        Compiler().compile(
            pipeline_func=training_only_pipeline,
            package_path=pipeline_json_path,
        )
        logger.info(f"Pipeline compilada exitosamente en '{pipeline_json_path}'.")

        logger.info("Lanzando la pipeline de entrenamiento en Vertex AI (modo síncrono)...")
        success = run_vertex_pipeline(
            pipeline_json_path=pipeline_json_path,
            project_id=constants.PROJECT_ID,
            region=constants.REGION,
            pipeline_root=constants.PIPELINE_ROOT,
            service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
            display_name_prefix="training-only-pipeline",
            enable_caching=True,
            pipeline_parameters=PIPELINE_PARAMETERS,
            sync=True # Esperar a que la pipeline termine
        )

        if not success:
            logger.error("La ejecución en Vertex AI falló. Abortando.")
            sys.exit(1) # Salir con código de error

    except Exception as e:
        logger.exception("❌ Fallo en el proceso de compilación o lanzamiento.")
        sys.exit(1)
    finally:
        if os.path.exists(pipeline_json_path):
            os.remove(pipeline_json_path)
            logger.info(f"Archivo de pipeline compilado '{pipeline_json_path}' eliminado.")

if __name__ == "__main__":
    main()