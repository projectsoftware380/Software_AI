# src/components/train_lstm_launcher/task.py (VERSIÓN CORREGIDA Y ROBUSTA)

from __future__ import annotations
import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform as aip
from src.shared import gcs_utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)
STAGING_BUCKET = "gs://trading-ai-models-460823/staging_for_custom_jobs"

# --------------------------------------------------------------------------- #
def run_launcher(
    *,
    project_id: str,
    region: str,
    pair: str,
    timeframe: str,
    params_path: str,
    features_gcs_path: str,
    output_gcs_base_dir: str,
    vertex_training_image_uri: str,
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
    trained_lstm_dir_path_output: str,
) -> None:

    aip.init(project=project_id, location=region)

    ts_disp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    display_name = f"train-lstm-{pair.lower()}-{timeframe.lower()}-{ts_disp}"

    # ---------- Localizar el best_params.json más reciente ----------
    best_params = gcs_utils.find_latest_gcs_file_in_timestamped_dirs(
        base_gcs_path=f"{params_path}/{pair}", filename="best_params.json"
    )
    if best_params is None:
        raise RuntimeError(f"No se encontró best_params.json en {params_path}/{pair}")
    logger.info("✔ Parámetros de trading encontrados: %s", best_params)

    # ---------- Definir el Custom Job de Vertex AI ----------
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": vertex_machine_type,
                "accelerator_type": vertex_accelerator_type,
                "accelerator_count": vertex_accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": vertex_training_image_uri,
                "command": ["python", "-m", "src.components.train_lstm.main"],
                "args": [
                    f"--pair={pair}",
                    f"--timeframe={timeframe}",
                    f"--params={best_params}",
                    f"--features-gcs-path={features_gcs_path}",
                    f"--output-gcs-base-dir={output_gcs_base_dir}",
                ],
            },
        }
    ]

    job = aip.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        # Dejamos que Vertex gestione la carpeta de salida para evitar conflictos.
        base_output_dir=None, 
        project=project_id,
        location=region,
        staging_bucket=STAGING_BUCKET,
    )

    logger.info("⏳ Lanzando Custom Job '%s' para entrenar el modelo LSTM...", display_name)
    job.run(service_account=vertex_service_account, sync=True)

    state = getattr(job.state, "name", str(job.state))
    if state != "JOB_STATE_SUCCEEDED":
        logger.error("❌ El Custom Job de entrenamiento LSTM falló. Revisa los logs en Vertex AI > Training.")
        sys.exit(1)

    # --- INICIO DE LA CORRECCIÓN CLAVE ---
    # Obtener la ruta de salida directamente del objeto del job, que es la fuente de verdad.
    # El script 'train_lstm.main' crea una subcarpeta con timestamp DENTRO de este directorio.
    
    job_output_dir = job.job_spec.base_output_directory
    logger.info("Directorio de salida base del job: %s", job_output_dir)

    # El script de entrenamiento crea una única carpeta con timestamp dentro del directorio base.
    # La buscamos para obtener la ruta final y precisa del modelo.
    latest_model_file = gcs_utils.find_latest_gcs_file_in_timestamped_dirs(
        base_gcs_path=f"{job_output_dir}/{pair}/{timeframe}",
        filename="model.keras"
    )

    if latest_model_file is None:
        raise RuntimeError(
            "El Custom Job terminó pero no se encontró 'model.keras' en la carpeta de salida esperada."
        )

    # Extraer solo la ruta del directorio
    final_model_path = latest_model_file.rsplit("/", 1)[0]
    logger.info("✅ Carpeta final del modelo identificada: %s", final_model_path)
    # --- FIN DE LA CORRECCIÓN CLAVE ---

    # Propagar la ruta correcta al siguiente paso de la pipeline
    dest = Path(trained_lstm_dir_path_output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(final_model_path, encoding="utf-8")
    logger.info("🔗 Ruta del modelo '%s' propagada con éxito a la pipeline.", final_model_path)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--params-path", required=True)
    p.add_argument("--features-gcs-path", required=True)
    p.add_argument("--output-gcs-base-dir", required=True)
    p.add_argument("--vertex-training-image-uri", required=True)
    p.add_argument("--vertex-machine-type", required=True)
    p.add_argument("--vertex-accelerator-type", required=True)
    p.add_argument("--vertex-accelerator-count", type=int, required=True)
    p.add_argument("--vertex-service-account", required=True)
    p.add_argument("--trained-lstm-dir-path-output", required=True)
    args = p.parse_args()

    run_launcher(
        project_id=args.project_id,
        region=args.region,
        pair=args.pair,
        timeframe=args.timeframe,
        params_path=args.params_path,
        features_gcs_path=args.features_gcs_path,
        output_gcs_base_dir=args.output_gcs_base_dir,
        vertex_training_image_uri=args.vertex_training_image_uri,
        vertex_machine_type=args.vertex_machine_type,
        vertex_accelerator_type=args.vertex_accelerator_type,
        vertex_accelerator_count=args.vertex_accelerator_count,
        vertex_service_account=args.vertex_service_account,
        trained_lstm_dir_path_output=args.trained_lstm_dir_path_output,
    )