# src/components/train_lstm_launcher/task.py
"""
Lanza un Custom Job de Vertex AI para entrenar el modelo LSTM
y escribe en disco la ruta final para que Kubeflow la propague.
"""

from __future__ import annotations
import argparse, logging, os, sys
from datetime import datetime, timezone
from pathlib import Path
from google.cloud import aiplatform as aip

# Importar las utilidades GCS para la nueva función
from src.shared import gcs_utils

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

STAGING_BUCKET = "gs://trading-ai-models-460823/staging_for_custom_jobs"

def run_launcher(*, project_id:str, region:str, pair:str, timeframe:str,
                 params_path:str, features_gcs_path:str, output_gcs_base_dir:str,
                 vertex_training_image_uri:str, vertex_machine_type:str,
                 vertex_accelerator_type:str, vertex_accelerator_count:int,
                 vertex_service_account:str, trained_lstm_dir_path_output:str) -> None:

    aip.init(project=project_id, location=region)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"train-lstm-{pair.lower()}-{timeframe.lower()}-{ts}"
    out  = os.path.join(output_gcs_base_dir, name)
    logger.info("⏳ Creando CustomJob %s", name)

    # MODIFICACIÓN: Obtener la ruta de parámetros más reciente usando la nueva utilidad
    # `params_path` que viene de la pipeline es ahora la base de directorios con timestamp
    actual_params_path = gcs_utils.find_latest_gcs_file_in_timestamped_dirs(
        base_gcs_path=f"{params_path}/{pair}", # Asumimos params_path es algo como 'gs://.../LSTM_v3'
        filename="best_params.json"
    )
    
    if actual_params_path is None:
        raise RuntimeError(f"No se encontró el archivo best_params.json para el par {pair} en la ruta {params_path}")

    logger.info(f"✔ Usando archivo de parámetros: {actual_params_path}")

    worker_pool_specs = [{
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
                f"--params={actual_params_path}", # USAR LA RUTA OBTENIDA
                f"--features-gcs-path={features_gcs_path}",
                f"--output-gcs-base-dir={output_gcs_base_dir}",
            ],
        },
    }]

    job = aip.CustomJob(
        display_name=name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=out,
        project=project_id,
        location=region,
        staging_bucket=STAGING_BUCKET,
    )

    job.run(service_account=vertex_service_account, sync=True)
    state = getattr(job.state, "name", str(job.state))
    logger.info("🏁 Estado final: %s", state)

    if state != "JOB_STATE_SUCCEEDED":
        logger.error("CustomJob falló — revisa Vertex AI → Jobs")
        sys.exit(1)

    dest = Path(trained_lstm_dir_path_output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out, encoding="utf-8")
    logger.info("🔗 Ruta del modelo guardada en %s", dest)

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
    p.add_argument("--vertex-accelerator-type", required=True) # Argumento definido correctamente
    p.add_argument("--vertex-accelerator-count", type=int, required=True)
    p.add_argument("--vertex-service-account", required=True)
    p.add_argument("--trained-lstm-dir-path-output", required=True)
    a = p.parse_args()

    run_launcher(
        project_id=a.project_id, region=a.region,
        pair=a.pair, timeframe=a.timeframe,
        params_path=a.params_path, features_gcs_path=a.features_gcs_path,
        output_gcs_base_dir=a.output_gcs_base_dir,
        vertex_training_image_uri=a.vertex_training_image_uri,
        vertex_machine_type=a.vertex_machine_type,
        vertex_accelerator_type=a.vertex_accelerator_type, # <-- CORRECCIÓN AQUÍ
        vertex_accelerator_count=a.vertex_accelerator_count,
        vertex_service_account=a.vertex_service_account,
        trained_lstm_dir_path_output=a.trained_lstm_dir_path_output,
    )