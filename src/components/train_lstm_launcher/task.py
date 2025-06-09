# src/components/train_lstm_launcher/task.py
"""
Lanzador del entrenamiento LSTM (v3) en Vertex AI Custom Job.

Flujo:
1. Inicializa Vertex AI.
2. Construye un Custom Job que lanza la imagen runner con los argumentos
   necesarios para entrenar el LSTM.
3. Bloquea (sync=True) hasta que el job termina.
4. Busca la carpeta GCS donde se guardó el modelo (.h5) y la expone:

   • Como valor de retorno de run_launcher().
   • Escribiéndola en '/tmp/trained_dir.txt' para que KFP la capture.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from google.cloud import aiplatform as gcp_aiplatform
from google.cloud import storage as gcp_storage

from src.shared import constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Función principal que envía el Custom Job y devuelve la URI final.
# ────────────────────────────────────────────────────────────────────
def run_launcher(
    *,
    project_id: str,
    region: str,
    pair: str,
    timeframe: str,
    params_path: str,
    output_gcs_base_dir: str,
    vertex_training_image_uri: str,
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
) -> str:
    """Envía el Custom Job y devuelve la carpeta GCS donde quedó el modelo."""

    # 1) Inicializar Vertex AI
    gcp_aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=constants.STAGING_PATH,
    )
    logger.info("Vertex AI inicializado · staging bucket: %s", constants.STAGING_PATH)

    # 2) Construir nombre y lista de argumentos para el contenedor
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_display_name = f"lstm-train-v3-{pair.lower()}-{timeframe.lower()}-{ts}"

    training_args: List[str] = [
        "python",
        "-m",
        "src.components.train_lstm.main",
        "--params",
        params_path,
        "--output-gcs-base-dir",
        output_gcs_base_dir,
        "--pair",
        pair,
        "--timeframe",
        timeframe,
    ]

    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": vertex_machine_type},
            "replica_count": 1,
            "container_spec": {
                "image_uri": vertex_training_image_uri,
                "args": training_args,
            },
        }
    ]

    # Añadir GPU si procede
    if vertex_accelerator_count and vertex_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        spec = worker_pool_specs[0]["machine_spec"]
        spec["accelerator_type"] = vertex_accelerator_type
        spec["accelerator_count"] = vertex_accelerator_count
        logger.info("Añadiendo GPU %s × %d", vertex_accelerator_type, vertex_accelerator_count)

    logger.info("Enviando Custom Job: %s", job_display_name)

    custom_job = gcp_aiplatform.CustomJob(
        display_name=job_display_name,
        project=project_id,
        location=region,
        worker_pool_specs=worker_pool_specs,
    )

    try:
        custom_job.run(
            service_account=vertex_service_account,
            sync=True,
            timeout=3 * 60 * 60,  # 3 h
        )
        logger.info("Custom Job %s completado ✔️", job_display_name)
    except Exception as err:  # pylint: disable=broad-except
        logger.error("Custom Job %s falló: %s", job_display_name, err, exc_info=True)
        raise RuntimeError(f"Vertex AI Custom Job falló: {err}") from err

    # 3) Localizar la carpeta del modelo
    bucket_name = constants.GCS_BUCKET_NAME
    prefix = (
        f"{output_gcs_base_dir.removeprefix(f'gs://{bucket_name}/')}"
        f"/{pair}/{timeframe}/"
    )

    storage_client = gcp_storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    # Esperar brevemente por consistencia
    time.sleep(10)

    candidate_dirs: list[str] = []
    for page in bucket.list_blobs(prefix=prefix, delimiter="/").pages:
        for dir_prefix in getattr(page, "prefixes", []):
            blob = bucket.blob(f"{dir_prefix.rstrip('/')}/model.h5")
            if blob.exists():
                candidate_dirs.append(f"gs://{bucket_name}/{dir_prefix.rstrip('/')}")

    if not candidate_dirs:
        raise TimeoutError(f"No se encontró model.h5 en gs://{bucket_name}/{prefix}")

    latest_model_dir = max(candidate_dirs)  # la de timestamp más reciente
    logger.info("Modelo encontrado en: %s", latest_model_dir)
    return latest_model_dir


# ────────────────────────────────────────────────────────────────────
# CLI – se usa tanto desde el componente KFP como manualmente
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Launcher Vertex AI LSTM Trainer")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--output-gcs-base-dir", required=True)
    parser.add_argument("--vertex-training-image-uri", required=True)
    parser.add_argument("--vertex-machine-type", required=True)
    parser.add_argument("--vertex-accelerator-type", required=True)
    parser.add_argument("--vertex-accelerator-count", type=int, required=True)
    parser.add_argument("--vertex-service-account", required=True)
    cli = parser.parse_args()

    trained_dir = run_launcher(
        project_id=cli.project_id,
        region=cli.region,
        pair=cli.pair,
        timeframe=cli.timeframe,
        params_path=cli.params_path,
        output_gcs_base_dir=cli.output_gcs_base_dir,
        vertex_training_image_uri=cli.vertex_training_image_uri,
        vertex_machine_type=cli.vertex_machine_type,
        vertex_accelerator_type=cli.vertex_accelerator_type,
        vertex_accelerator_count=cli.vertex_accelerator_count,
        vertex_service_account=cli.vertex_service_account,
    )

    # Exportar la ruta para KFP (fileOutputs)
    Path("/tmp/trained_dir.txt").write_text(trained_dir)
    print(f"Trained model artifacts directory: {trained_dir}")
