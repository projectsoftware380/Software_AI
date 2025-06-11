"""
Lanza un Custom Job de Vertex AI para entrenar el modelo LSTM y
escribe en disco la ruta de los artefactos.  *Necesita Python 3.8+*
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from google.cloud import aiplatform as aip

# ——— logging ————————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ——— bucket de staging ————————————————————————————————
STAGING_BUCKET = "gs://trading-ai-models-460823/staging_for_custom_jobs"

# ───────────────────────── función principal ─────────────────────────
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
    """Ejecuta el CustomJob y guarda la ruta de salida para KFP."""
    aip.init(project=project_id, location=region)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_display_name = f"train-lstm-{pair.lower()}-{timeframe.lower()}-{ts}"
    output_dir = os.path.join(output_gcs_base_dir, job_display_name)
    logger.info("⏳ Creando CustomJob %s", job_display_name)

    # ——— definición del worker ——————————————————————
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
                    f"--params={params_path}",                 # ✅ nombre correcto
                    f"--features-gcs-path={features_gcs_path}",
                    f"--output-gcs-base-dir={output_gcs_base_dir}",  # ✅ nombre correcto
                ],
            },
        }
    ]

    # ——— CustomJob con staging_bucket ————————————————
    job = aip.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=output_dir,
        project=project_id,
        location=region,
        staging_bucket=STAGING_BUCKET,
    )

    job.run(service_account=vertex_service_account, sync=True)
    logger.info("🏁 Estado final: %s", job.state.name)

    if job.state != aip.JobState.JOB_STATE_SUCCEEDED:
        logger.error("El entrenamiento falló — revisa Vertex AI → Jobs.")
        sys.exit(1)

    # ——— informar a KFP ——————————————————————————
    logger.info("🔗 Guardando ruta del modelo entrenado en %s", trained_lstm_dir_path_output)
    with open(trained_lstm_dir_path_output, "w", encoding="utf-8") as f:
        f.write(output_dir)

# ──────────────────────────── CLI / Entrypoint ────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--params-path", required=True)           # recibe nombre viejo desde KFP
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
