# src/components/train_lstm_launcher/task.py

from __future__ import annotations
import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform as aip

# Se importan las constantes y utilidades de GCS para una operación robusta.
from src.shared import constants, gcs_utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
def run_launcher(
    *,
    project_id: str,
    region: str,
    pair: str,
    timeframe: str,
    params_file: str,  # <-- CORRECCIÓN: Ahora se recibe la ruta exacta del archivo.
    features_gcs_path: str,
    output_gcs_base_dir: str,
    vertex_training_image_uri: str,
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
    trained_lstm_dir_path_output: str,
) -> None:
    """Orquesta el lanzamiento de un CustomJob en Vertex AI para el entrenamiento."""

    aip.init(project=project_id, location=region, staging_bucket=constants.STAGING_PATH)

    ts_disp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    display_name = f"train-lstm-{pair.lower()}-{timeframe.lower()}-{ts_disp}"

    # Se construye la ruta de salida final y única para este trabajo.
    final_model_output_dir = f"{output_gcs_base_dir}/{pair}/{timeframe}/{display_name}"
    logger.info("Ruta de salida final del modelo definida: %s", final_model_output_dir)

    # Ya no es necesario buscar el archivo, se usa la ruta exacta proporcionada.
    logger.info("✔ Usando el archivo de parámetros de trading de: %s", params_file)

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
                    # --- CORRECCIÓN ---
                    # Se pasa la ruta exacta del archivo al script de entrenamiento.
                    f"--params-file={params_file}",
                    f"--features-gcs-path={features_gcs_path}",
                    f"--output-gcs-final-dir={final_model_output_dir}",
                ],
            },
        }
    ]

    job = aip.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        project=project_id,
        location=region,
    )

    logger.info("⏳ Lanzando Custom Job '%s' para entrenar el modelo LSTM...", display_name)
    job.run(service_account=vertex_service_account, sync=True)

    state = getattr(job.state, "name", str(job.state))
    if state != "JOB_STATE_SUCCEEDED":
        logger.error("❌ El Custom Job de entrenamiento LSTM falló. Revisa los logs en Vertex AI > Training.")
        sys.exit(1)

    logger.info("✅ El Custom Job de entrenamiento finalizó con éxito.")
    
    # Se propaga la ruta conocida y exacta al siguiente paso de la pipeline.
    dest = Path(trained_lstm_dir_path_output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(final_model_output_dir, encoding="utf-8")
    logger.info("🔗 Ruta del modelo '%s' propagada con éxito a la pipeline.", final_model_output_dir)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Lanzador de trabajos de entrenamiento LSTM en Vertex AI.")
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    # --- CORRECCIÓN: El argumento ahora es `--params-file` para reflejar que es una ruta de archivo. ---
    p.add_argument("--params-file", required=True)
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
        params_file=args.params_file,
        features_gcs_path=args.features_gcs_path,
        output_gcs_base_dir=args.output_gcs_base_dir,
        vertex_training_image_uri=args.vertex_training_image_uri,
        vertex_machine_type=args.vertex_machine_type,
        vertex_accelerator_type=args.vertex_accelerator_type,
        vertex_accelerator_count=args.vertex_accelerator_count,
        vertex_service_account=args.vertex_service_account,
        trained_lstm_dir_path_output=args.trained_lstm_dir_path_output,
    )