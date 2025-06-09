# src/components/train_lstm_launcher/task.py
"""
Tarea del componente LANZADOR de entrenamiento del modelo LSTM.

Responsabilidades:
1.  Configurar y lanzar un Vertex AI Custom Job para el entrenamiento.
2.  Pasar todos los parámetros necesarios (rutas, hiperparámetros, configuración
    de hardware) al Custom Job.
3.  El Custom Job ejecutará un script de entrenamiento separado (contenido en la
    imagen `runner-lstm`) en hardware dedicado (potencialmente con GPUs).
4.  Esperar a que el Custom Job termine.
5.  Una vez terminado, buscar en GCS el directorio de salida versionado que
    el job de entrenamiento ha creado.
6.  Devolver la ruta GCS a este directorio para que los siguientes pasos de la
    pipeline puedan usar los artefactos del modelo (modelo, scaler, params).

Este script se ejecuta en un pod normal de KFP, NO en el hardware de entrenamiento.
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

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s [%(funcName)s]: %(message)s",
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
    """Orquesta el lanzamiento y monitoreo del Vertex AI Custom Job."""
    
    # 1. Inicializar Vertex AI SDK
    staging_gcs_path = constants.STAGING_PATH
    gcp_aiplatform.init(
        project=project_id, location=region, staging_bucket=staging_gcs_path
    )
    logger.info(f"Vertex AI SDK inicializado. Staging en: {staging_gcs_path}")

    # 2. Configurar el Custom Job
    timestamp_for_job = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_display_name = f"lstm-train-v3-{pair.lower()}-{timeframe.lower()}-{timestamp_for_job}"

    # Argumentos que se pasarán al script *dentro* del contenedor del Custom Job
    training_script_args = [
        "--params", params_path,
        "--output-gcs-base-dir", output_gcs_base_dir,
        "--pair", pair,
        "--timeframe", timeframe,
    ]

    worker_pool_specs = [{
        "machine_spec": { "machine_type": vertex_machine_type },
        "replica_count": 1,
        "container_spec": {
            "image_uri": vertex_training_image_uri,
            "args": training_script_args,
        },
    }]

    if vertex_accelerator_count > 0 and vertex_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        worker_pool_specs[0]["machine_spec"]["accelerator_type"] = vertex_accelerator_type
        worker_pool_specs[0]["machine_spec"]["accelerator_count"] = vertex_accelerator_count
        logger.info(f"Configurando Custom Job con GPU: {vertex_accelerator_count}x{vertex_accelerator_type}")

    # 3. Lanzar el job
    logger.info(f"Enviando Vertex AI Custom Job: {job_display_name}")
    logger.info(f"  Worker pool specs: {json.dumps(worker_pool_specs)}")

    custom_job = gcp_aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
        project=project_id,
        location=region,
    )
    
    # Se usa .run() para una ejecución síncrona. La pipeline esperará aquí.
    try:
        custom_job.run(service_account=vertex_service_account, sync=True, timeout=10800) # 3 horas timeout
        logger.info(f"✅ Vertex AI Custom Job {custom_job.display_name} completado.")
    except Exception as e:
        logger.error(f"❌ Vertex AI Custom Job {job_display_name} falló: {e}")
        if custom_job and custom_job.resource_name:
            logger.error(f"  Detalles del Job: {custom_job.resource_name}, estado: {custom_job.state}")
        raise RuntimeError(f"Vertex AI Custom Job para entrenamiento LSTM falló: {e}")

    # 4. Encontrar el directorio de salida del modelo
    # El script de entrenamiento crea un subdirectorio con timestamp.
    # Necesitamos encontrarlo para pasarlo al siguiente paso.
    bucket_name, _ = constants.GCS_BUCKET_NAME, constants.LSTM_MODELS_PATH.split(f"gs://{constants.GCS_BUCKET_NAME}/")[1]
    gcs_listing_prefix = f"{output_gcs_base_dir.replace(f'gs://{bucket_name}/', '')}/{pair}/{timeframe}/"

    logger.info(f"Buscando directorio de salida del modelo en gs://{bucket_name}/{gcs_listing_prefix}...")

    storage_client = gcp_storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Esperar un poco por si hay latencia en GCS
    time.sleep(10) 
    
    blobs = bucket.list_blobs(prefix=gcs_listing_prefix, delimiter="/")
    candidate_dirs = []
    for page in blobs.pages:
        # page.prefixes contiene los "directorios"
        if hasattr(page, 'prefixes') and page.prefixes:
            for dir_prefix in page.prefixes:
                # Verificar si contiene el archivo model.h5 para confirmar que es un dir de modelo
                model_blob_path = f"{dir_prefix.rstrip('/')}/model.h5"
                if bucket.blob(model_blob_path).exists():
                    candidate_dirs.append(f"gs://{bucket_name}/{dir_prefix.rstrip('/')}")
    
    if not candidate_dirs:
        err_msg = f"No se encontró el directorio de salida del modelo LSTM en gs://{bucket_name}/{gcs_listing_prefix}."
        logger.error(err_msg)
        raise TimeoutError(err_msg)

    # Devolver la ruta más reciente
    latest_model_dir = sorted(candidate_dirs, reverse=True)[0]
    logger.info(f"🎉 Tarea completada. Directorio del modelo encontrado: {latest_model_dir}")
    return latest_model_dir

# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task Lanzador de Entrenamiento LSTM en Vertex AI.")
    
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
    parser.add_argument("--trained-lstm-dir-path-output", type=Path, required=True)
    
    args = parser.parse_args()

    trained_model_path = run_launcher(
        project_id=args.project_id,
        region=args.region,
        pair=args.pair,
        timeframe=args.timeframe,
        params_path=args.params_path,
        output_gcs_base_dir=args.output_gcs_base_dir,
        vertex_training_image_uri=args.vertex_training_image_uri,
        vertex_machine_type=args.vertex_machine_type,
        vertex_accelerator_type=args.vertex_accelerator_type,
        vertex_accelerator_count=args.vertex_accelerator_count,
        vertex_service_account=args.vertex_service_account,
    )

    # Escribir la ruta de salida al archivo que KFP espera
    args.trained_lstm_dir_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.trained_lstm_dir_path_output.write_text(trained_model_path)