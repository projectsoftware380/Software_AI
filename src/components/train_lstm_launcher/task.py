# src/components/train_lstm_launcher/task.py
"""
Lanza un Custom Job en Vertex AI que entrena el LSTM y escribe la ruta
final del modelo en un archivo de salida para Kubeflow.

Corrección 2025-06-14 — Se unifica el formato del timestamp a 14 dígitos
(YYYYMMDDHHMMSS) para que coincida con la carpeta generada por
src.components.train_lstm.main y evitar errores 404 en train_filter_model.
"""

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
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

STAGING_BUCKET = "gs://trading-ai-models-460823/staging_for_custom_jobs"


# --------------------------------------------------------------------------- #
#  FUNCIÓN PRINCIPAL                                                          #
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
    """Crea y ejecuta el Custom Job que entrena el LSTM."""

    aip.init(project=project_id, location=region)

    # Timestamp UNIFICADO → 14 dígitos sin guion (YYYYMMDDHHMMSS)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    # Nombre visible del job (puede llevar guion o no; aquí sin guion)
    display_name = f"train-lstm-{pair.lower()}-{timeframe.lower()}-{ts}"

    # Carpeta definitiva donde se guardará el modelo
    # Coincide 100 % con la ruta usada por src.components.train_lstm.main
    model_dir = f"{output_gcs_base_dir}/{pair}/{timeframe}/{ts}"
    logger.info("El modelo se guardará en: %s", model_dir)

    # ------------------------------------------------------------------ #
    #  Localizar el best_params.json más reciente                        #
    # ------------------------------------------------------------------ #
    best_params = gcs_utils.find_latest_gcs_file_in_timestamped_dirs(
        base_gcs_path=f"{params_path}/{pair}",
        filename="best_params.json",
    )
    if best_params is None:
        raise RuntimeError(
            f"No se encontró best_params.json en {params_path}/{pair}"
        )
    logger.info("✔ Parámetros: %s", best_params)

    # ------------------------------------------------------------------ #
    #  Definición del Custom Job                                         #
    # ------------------------------------------------------------------ #
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
        base_output_dir=model_dir,  # Directorio unificado sin guion
        project=project_id,
        location=region,
        staging_bucket=STAGING_BUCKET,
    )

    logger.info("⏳ Lanzando Custom Job %s", display_name)
    job.run(service_account=vertex_service_account, sync=True)

    state = getattr(job.state, "name", str(job.state))
    logger.info("🏁 Estado final: %s", state)
    if state != "JOB_STATE_SUCCEEDED":
        logger.error("Custom Job falló — revisa Vertex AI → Training")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Propagar la ruta para el siguiente paso de la pipeline            #
    # ------------------------------------------------------------------ #
    dest = Path(trained_lstm_dir_path_output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(model_dir, encoding="utf-8")
    logger.info("🔗 Ruta del modelo guardada en %s", dest)


# --------------------------------------------------------------------------- #
#  ENTRYPOINT CLI                                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--features-gcs-path", required=True)
    parser.add_argument("--output-gcs-base-dir", required=True)
    parser.add_argument("--vertex-training-image-uri", required=True)
    parser.add_argument("--vertex-machine-type", required=True)
    parser.add_argument("--vertex-accelerator-type", required=True)
    parser.add_argument("--vertex-accelerator-count", type=int, required=True)
    parser.add_argument("--vertex-service-account", required=True)
    parser.add_argument("--trained-lstm-dir-path-output", required=True)
    args = parser.parse_args()

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
