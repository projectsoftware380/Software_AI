# RUTA: src/components/train_lstm_launcher/task.py
# DESCRIPCIÓN: Versión corregida que pasa la ruta de datos al CustomJob.

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from google.cloud import aiplatform as gcp_aiplatform
from google.cloud import storage as gcp_storage

from src.shared import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def run_launcher(
    *,
    project_id: str,
    region: str,
    pair: str,
    timeframe: str,
    params_path: str,
    features_gcs_path: str, # <-- AJUSTE: Añadir parámetro
    output_gcs_base_dir: str,
    vertex_training_image_uri: str,
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
) -> str:
    gcp_aiplatform.init(project=project_id, location=region, staging_bucket=constants.STAGING_PATH)
    logger.info("Vertex AI inicializado · staging bucket: %s", constants.STAGING_PATH)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_display_name = f"lstm-train-v3-{pair.lower()}-{timeframe.lower()}-{ts}"

    # AJUSTE: Añadir el nuevo argumento a la lista de args que se pasarán al contenedor
    container_args = [
        "--params", params_path,
        "--features-gcs-path", features_gcs_path,
        "--output-gcs-base-dir", output_gcs_base_dir,
        "--pair", pair,
        "--timeframe", timeframe,
    ]

    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": vertex_machine_type},
            "replica_count": 1,
            "container_spec": {
                "image_uri": vertex_training_image_uri,
                "command": ["python", "-m", "src.components.train_lstm.main"],
                "args": container_args, # <-- Usar la lista de args actualizada
            },
        }
    ]

    if vertex_accelerator_count > 0 and vertex_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        spec = worker_pool_specs[0]["machine_spec"]
        spec["accelerator_type"] = vertex_accelerator_type
        spec["accelerator_count"] = vertex_accelerator_count
        logger.info("Añadiendo GPU %s × %d", vertex_accelerator_type, vertex_accelerator_count)

    logger.info("Enviando Custom Job: %s con args: %s", job_display_name, container_args)

    custom_job = gcp_aiplatform.CustomJob(
        display_name=job_display_name,
        project=project_id,
        location=region,
        worker_pool_specs=worker_pool_specs,
    )

    try:
        custom_job.run(service_account=vertex_service_account, sync=True, timeout=3 * 60 * 60)
        logger.info("Custom Job %s completado ✔️", job_display_name)
    except Exception as err:
        logger.error("Custom Job %s falló: %s", job_display_name, err, exc_info=True)
        raise RuntimeError(f"Vertex AI Custom Job falló: {err}") from err

    # ... (La lógica para encontrar la carpeta del modelo permanece IGUAL) ...
    # ... (Copia y pega el resto de tu función run_launcher aquí) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Launcher Vertex AI LSTM Trainer")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--params-path", required=True)
    # --- AJUSTE: Añadir el nuevo argumento al parser ---
    parser.add_argument("--features-gcs-path", required=True)
    # ---------------------------------------------------
    parser.add_argument("--output-gcs-base-dir", required=True)
    parser.add_argument("--vertex-training-image-uri", required=True)
    parser.add_argument("--vertex-machine-type", required=True)
    parser.add_argument("--vertex-accelerator-type", required=True)
    parser.add_argument("--vertex-accelerator-count", type=int, required=True)
    parser.add_argument("--vertex-service-account", required=True)
    cli = parser.parse_args()

    # --- AJUSTE: Pasar el nuevo argumento a la función ---
    trained_dir = run_launcher(
        project_id=cli.project_id,
        region=cli.region,
        pair=cli.pair,
        timeframe=cli.timeframe,
        params_path=cli.params_path,
        features_gcs_path=cli.features_gcs_path,
        output_gcs_base_dir=cli.output_gcs_base_dir,
        vertex_training_image_uri=cli.vertex_training_image_uri,
        vertex_machine_type=cli.vertex_machine_type,
        vertex_accelerator_type=cli.vertex_accelerator_type,
        vertex_accelerator_count=cli.vertex_accelerator_count,
        vertex_service_account=cli.vertex_service_account,
    )
    # --------------------------------------------------

    Path("/tmp/trained_dir.txt").write_text(trained_dir)
    print(f"Trained model artifacts directory: {trained_dir}")