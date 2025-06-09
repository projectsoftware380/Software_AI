# Lanzador del entrenamiento final LSTM en Vertex AI
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


def run_launcher(
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
    """
    Lanza un Vertex AI Custom Job que corre el módulo de entrenamiento LSTM.
    Devuelve la ruta GCS donde se guardó el modelo entrenado.
    """

    # ── 1 Inicializar Vertex AI ─────────────────────────────
    gcp_aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=constants.STAGING_PATH,
    )
    logger.info("Vertex AI inicializado (staging bucket: %s).", constants.STAGING_PATH)

    # ── 2 Definir el Custom Job ────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_display_name = f"lstm-train-v3-{pair.lower()}-{timeframe.lower()}-{ts}"

    # 🔑 Cambio clave: invocar el módulo con python -m
    training_script_args = [
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
                "args": training_script_args,
            },
        }
    ]

    if (
        vertex_accelerator_count > 0
        and vertex_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED"
    ):
        spec = worker_pool_specs[0]["machine_spec"]
        spec["accelerator_type"] = vertex_accelerator_type
        spec["accelerator_count"] = vertex_accelerator_count
        logger.info(
            "Añadiendo GPU %s × %d al Custom Job.",
            vertex_accelerator_type,
            vertex_accelerator_count,
        )

    logger.info("Enviando Custom Job Vertex AI: %s", job_display_name)
    logger.debug("Worker Pool Specs: %s", json.dumps(worker_pool_specs, indent=2))

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
        if custom_job.resource_name:
            logger.error("Detalles: %s | state=%s", custom_job.resource_name, custom_job.state)
        raise RuntimeError(f"Vertex AI Custom Job falló: {err}") from err

    # ── 3 Localizar la carpeta del modelo en GCS ───────────
    bucket_name = constants.GCS_BUCKET_NAME
    prefix = f"{output_gcs_base_dir.removeprefix(f'gs://{bucket_name}/')}/{pair}/{timeframe}/"

    logger.info("Buscando artefactos en gs://%s/%s …", bucket_name, prefix)
    time.sleep(10)  # pequeña espera por consistencia

    storage_client = gcp_storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    candidate_dirs: list[str] = []
    for page in bucket.list_blobs(prefix=prefix, delimiter="/").pages:
        if getattr(page, "prefixes", None):
            for dir_prefix in page.prefixes:
                if bucket.blob(f"{dir_prefix.rstrip('/')}/model.h5").exists():
                    candidate_dirs.append(f"gs://{bucket_name}/{dir_prefix.rstrip('/')}")

    if not candidate_dirs:
        raise TimeoutError(
            f"No se encontró el modelo en gs://{bucket_name}/{prefix}"
        )

    latest_model_dir = max(candidate_dirs)  # el timestamp más reciente
    logger.info("Modelo encontrado en: %s", latest_model_dir)
    return latest_model_dir


# ──────────────────── CLI ──────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Launcher Vertex AI LSTM Trainer")
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--params-path", required=True)
    p.add_argument("--output-gcs-base-dir", required=True)
    p.add_argument("--vertex-training-image-uri", required=True)
    p.add_argument("--vertex-machine-type", required=True)
    p.add_argument("--vertex-accelerator-type", required=True)
    p.add_argument("--vertex-accelerator-count", type=int, required=True)
    p.add_argument("--vertex-service-account", required=True)
    p.add_argument("--trained-lstm-dir-path-output", type=Path, required=True)
    cli = p.parse_args()

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

    cli.trained_lstm_dir_path_output.parent.mkdir(parents=True, exist_ok=True)
    cli.trained_lstm_dir_path_output.write_text(trained_dir)
