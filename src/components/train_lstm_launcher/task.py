# RUTA: src/components/train_lstm_launcher/task.py
# DESCRIPCIÓN: Lanza un Vertex AI Custom Job para entrenar el modelo LSTM y
#              devuelve en un archivo la carpeta GCS donde se guardaron
#              los artefactos entrenados, para que KFP la encadene.

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from google.cloud import aiplatform as aip

# ――― Configuración de logging ――― #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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
    trained_lstm_dir_path_output: str,  # archivo de salida que KFP crea
) -> None:
    """Crea y ejecuta un Custom Job; escribe la ruta resultante en el
    archivo que KFP indica mediante --trained-lstm-dir-path-output."""

    aip.init(project=project_id, location=region)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_display_name = f"train-lstm-{pair.lower()}-{timeframe.lower()}-{timestamp}"
    logger.info(f"⏳ Lanzando CustomJob: {job_display_name}")

    # Ruta donde el job grabará modelos y métricas
    output_dir = os.path.join(output_gcs_base_dir, job_display_name)

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
                    f"--params-path={params_path}",
                    f"--features-gcs-path={features_gcs_path}",
                    f"--output-gcs-dir={output_dir}",
                ],
            },
        }
    ]

    job = aip.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
        project=project_id,
        location=region,
        base_output_dir=output_gcs_base_dir,
    )

    job.run(service_account=vertex_service_account, sync=True)
    logger.info(f"🏁 Estado final del CustomJob: {job.state}")

    if job.state != aip.JobState.JOB_STATE_SUCCEEDED:
        logger.error("El entrenamiento falló - revisa la consola de Vertex AI.")
        sys.exit(1)

    # ――― Escribir la salida para KFP ――― #
    logger.info(f"🔗 Guardando ruta del modelo entrenado: {output_dir}")
    with open(trained_lstm_dir_path_output, "w") as f:
        f.write(output_dir)
    logger.info("✅ Ruta escrita correctamente")


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
    # ← argumento que KFP rellena automáticamente con un archivo temporal
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
