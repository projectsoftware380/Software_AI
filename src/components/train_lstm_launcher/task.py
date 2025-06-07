# src/components/train_lstm_launcher/task.py (VERSIÓN COMPLETA Y CORREGIDA)
"""
Tarea del componente lanzador del entrenamiento en Vertex AI.
Este script se ejecuta en KFP para crear y monitorear un CustomJob en Vertex AI.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

from google.cloud import aiplatform as gcp_aiplatform
from google.cloud import storage as gcp_storage

# Es importante que este script pueda importar 'constants' si se ejecuta con 'python -m'
from src.shared import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_vertex_job(
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
    """Orquesta el lanzamiento y monitoreo del Vertex AI Custom Job."""
    
    gcs_bucket_name = output_gcs_base_dir.split('/')[2]
    staging_gcs_path = f"gs://{gcs_bucket_name}/staging_for_custom_jobs"
    
    gcp_aiplatform.init(project=project_id, location=region, staging_bucket=staging_gcs_path)
    logger.info(f"Vertex AI SDK inicializado. Proyecto: {project_id}, Región: {region}")

    timestamp_for_job = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_display_name = f"lstm-training-{pair.lower()}-{timeframe.lower()}-{timestamp_for_job}"

    # Argumentos que se pasarán al script DE ENTRENAMIENTO dentro del contenedor del Custom Job
    training_script_args = [
        "--params", params_path,
        "--output-gcs-base-dir", output_gcs_base_dir,
        "--pair", pair,
        "--timeframe", timeframe,
    ]

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": vertex_machine_type,
            "accelerator_type": vertex_accelerator_type,
            "accelerator_count": vertex_accelerator_count,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": vertex_training_image_uri,
            "args": training_script_args,
        },
    }]

    if vertex_accelerator_count > 0:
        logger.info(f"Configurando Custom Job con GPU: {vertex_accelerator_count}x{vertex_accelerator_type}")

    vertex_job_base_output_dir = f"gs://{gcs_bucket_name}/vertex_ai_job_outputs/{job_display_name}"

    logger.info(f"Enviando Vertex AI Custom Job: {job_display_name}")
    logger.info(f"  - Imagen Docker: {vertex_training_image_uri}")
    logger.info(f"  - Args para el script de entrenamiento: {json.dumps(training_script_args)}")

    custom_job = gcp_aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=vertex_job_base_output_dir,
    )

    try:
        custom_job.run(service_account=vertex_service_account, sync=True, timeout=10800)
        logger.info(f"✅ Vertex AI Custom Job {custom_job.display_name} completado.")
    except Exception as e:
        logger.error(f"❌ Vertex AI Custom Job {job_display_name} falló: {e}", exc_info=True)
        raise RuntimeError(f"Fallo en el entrenamiento LSTM en Vertex AI: {e}")

    # Sondeo para encontrar la ruta de salida exacta creada por el job
    prefix_parts = output_gcs_base_dir.replace("gs://", "").split("/")
    gcs_listing_prefix = f"{'/'.join(prefix_parts[1:])}/{pair}/{timeframe}/"
    
    logger.info(f"Buscando artefactos de modelo en gs://{gcs_bucket_name}/{gcs_listing_prefix}")
    storage_client = gcp_storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket_name)
    
    max_wait_sec, poll_interval_sec = 10 * 60, 20
    start_time = time.time()
    
    while time.time() - start_time < max_wait_sec:
        blobs = bucket.list_blobs(prefix=gcs_listing_prefix, delimiter="/")
        candidate_dirs = [p for page in blobs.pages for p in page.prefixes]
        
        if candidate_dirs:
            latest_dir = sorted(candidate_dirs, reverse=True)[0]
            model_artifact_path = f"gs://{gcs_bucket_name}/{latest_dir.rstrip('/')}"
            logger.info(f"🎉 Directorio de modelo LSTM encontrado: {model_artifact_path}")
            return model_artifact_path
        
        logger.info(f"Esperando por los artefactos del modelo... ({int(time.time() - start_time)}s)")
        time.sleep(poll_interval_sec)

    raise TimeoutError(f"No se encontraron los artefactos del modelo en {output_gcs_base_dir} después de {max_wait_sec/60:.1f} minutos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanzador de Vertex AI Custom Job para entrenamiento LSTM.")
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
    
    args = parser.parse_args()
    final_model_dir = run_vertex_job(**vars(args))
    print(final_model_dir)