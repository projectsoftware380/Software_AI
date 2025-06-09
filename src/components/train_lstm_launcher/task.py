"""Lanzador del entrenamiento LSTM (v3) en Vertex AI Custom Job.

Este script:
1. Inicializa Vertex AI.
2. Construye un Custom Job que arranca el contenedor declarado en
   `constants.VERTEX_LSTM_TRAINER_IMAGE_URI`, invocando el módulo
   `src.components.train_lstm.main` con los argumentos indicados.
3. Espera su finalización (sync=True).
4. Localiza la carpeta GCS donde se guardó el modelo (.h5) y la devuelve
   (o la escribe en un archivo, si se usa como componente KFP).

NOTA – Toda la infraestructura usa **una sola imagen**:
`europe-west1-docker.pkg.dev/<PROYECTO>/data-ingestion-repo/data-ingestion-agent:latest`
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform as gcp_aiplatform
from google.cloud import storage as gcp_storage

from src.shared import constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s [%(funcName)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────────────────────────────────────
def run_launcher(
    project_id: str,
    region: str,
    pair: str,
    timeframe: str,
    params_path: str,
    output_gcs_base_dir: str,
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
) -> str:
    """Envía el Custom Job y devuelve la carpeta GCS del modelo entrenado."""

    # 1) Inicializar Vertex AI
    gcp_aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=constants.STAGING_PATH,
    )
    logger.info("Vertex AI inicializado (staging bucket: %s).", constants.STAGING_PATH)

    # 2) Preparar nombre y argumentos
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_display_name = f"lstm-train-v3-{pair.lower()}-{timeframe.lower()}-{ts}"

    training_args = [
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
                "image_uri": constants.VERTEX_LSTM_TRAINER_IMAGE_URI,
                "args": training_args,
            },
        }
    ]

    if vertex_accelerator_count and vertex_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        spec = worker_pool_specs[0]["machine_spec"]
        spec["accelerator_type"] = vertex_accelerator_type
        spec["accelerator_count"] = vertex_accelerator_count
        logger.info(
            "Añadiendo GPU %s × %d.", vertex_accelerator_type, vertex_accelerator_count
        )

    logger.info("Enviando Custom Job: %s", job_display_name)
    logger.debug("Worker Pool Specs:\n%s", json.dumps(worker_pool_specs, indent=2))

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
    except Exception as err:
        logger.error("Custom Job %s falló: %s", job_display_name, err)
        raise RuntimeError(f"Vertex AI Custom Job falló: {err}") from err

    # 3) Localizar la carpeta del modelo
    bucket_name = constants.GCS_BUCKET_NAME
    prefix = (
        f"{output_gcs_base_dir.removeprefix(f'gs://{bucket_name}/')}"
        f"/{pair}/{timeframe}/"
    )

    storage_client = gcp_storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    # Pequeña espera para consistencia GCS
    time.sleep(10)

    candidate_dirs: list[str] = []
    for page in bucket.list_blobs(prefix=prefix, delimiter="/").pages:
        if getattr(page, "prefixes", None):
            for dir_prefix in page.prefixes:
                if bucket.blob(f"{dir_prefix.rstrip('/')}/model.h5").exists():
                    candidate_dirs.append(f"gs://{bucket_name}/{dir_prefix.rstrip('/')}")

    if not candidate_dirs:
        raise TimeoutError(f"No se encontró modelo en gs://{bucket_name}/{prefix}")

    latest_model_dir = max(candidate_dirs)  # timestamp más reciente
    logger.info("Modelo encontrado en: %s", latest_model_dir)
    return latest_model_dir


# ──────────────────────────────────────────────────────────────────────────
# CLI (para uso como componente KFP o manual)
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Launcher Vertex AI LSTM Trainer")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--output-gcs-base-dir", required=True)
    parser.add_argument("--vertex-machine-type", required=True)
    parser.add_argument("--vertex-accelerator-type", required=True)
    parser.add_argument("--vertex-accelerator-count", type=int, required=True)
    parser.add_argument("--vertex-service-account", required=True)
    parser.add_argument("--trained-lstm-dir-path-output", type=Path, required=True)
    cli = parser.parse_args()

    out_dir = run_launcher(
        project_id=cli.project_id,
        region=cli.region,
        pair=cli.pair,
        timeframe=cli.timeframe,
        params_path=cli.params_path,
        output_gcs_base_dir=cli.output_gcs_base_dir,
        vertex_machine_type=cli.vertex_machine_type,
        vertex_accelerator_type=cli.vertex_accelerator_type,
        vertex_accelerator_count=cli.vertex_accelerator_count,
        vertex_service_account=cli.vertex_service_account,
    )

    cli.trained_lstm_dir_path_output.parent.mkdir(parents=True, exist_ok=True)
    cli.trained_lstm_dir_path_output.write_text(out_dir)
