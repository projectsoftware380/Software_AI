from typing import NamedTuple, List, Dict
import google.cloud.aiplatform as aiplatform_sdk
from google.cloud import storage
from google.cloud import secretmanager # Para Secret Manager
import os
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

# KFP imports
from kfp import compiler, dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact #

# --- Global Project and Bucket Configuration ---
PROJECT_ID = "trading-ai-460823"
REGION = "us-central1"

GCS_BUCKET_NAME = "trading-ai-models-460823"
PIPELINE_ROOT = f"gs://{GCS_BUCKET_NAME}/pipeline_root_v2"

# --- Docker Image URIs ---
KFP_COMPONENTS_IMAGE_URI = (
    f"europe-west1-docker.pkg.dev/{PROJECT_ID}/data-ingestion-repo/data-ingestion-agent:latest"
)

# Imagen "runner" para el entrenamiento del modelo LSTM en Vertex AI
# Esta imagen debe tener gsutil y las dependencias de Python para train_lstm.py
# y un ENTRYPOINT (ej. run.sh) que descargue y ejecute train_lstm.py
VERTEX_LSTM_TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/trading-images/runner-lstm:latest"
)

# --- Machine Types and GPUs for Vertex AI ---
DEFAULT_VERTEX_LSTM_MACHINE_TYPE = "n1-standard-4"
DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT = 0

KFP_COMPONENT_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
KFP_COMPONENT_ACCELERATOR_COUNT = 0

# --- Service Account for Vertex AI Custom Jobs ---
DEFAULT_VERTEX_LSTM_SERVICE_ACCOUNT = "data-ingestion-agent@trading-ai-460823.iam.gserviceaccount.com"


# --- Default Pipeline Hyperparameters ---
DEFAULT_N_TRIALS = 2
DEFAULT_PAIR = "EURUSD"
DEFAULT_TIMEFRAME = "15minute"


# --- Component Definitions ---
@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
    packages_to_install=[
        "google-cloud-storage",
        "google-cloud-pubsub",
        "gcsfs",
        "pandas",
        "google-cloud-secret-manager",
        "requests",
    ]
)
def update_data_op(
    pair: str,
    timeframe: str,
    gcs_bucket_name: str,
    project_id: str,
    polygon_api_key_secret_name: str,
    polygon_api_key_secret_version: str = "latest",
) -> str:
    import subprocess
    import logging
    import json
    import os
    import requests
    from google.cloud import secretmanager as sm_client_lib

    # Configurar logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s [%(funcName)s]: %(message)s")
    logger = logging.getLogger(__name__)

    try:
        metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"
        headers = {"Metadata-Flavor": "Google"}
        sa_email_response = requests.get(metadata_url, headers=headers, timeout=5)
        sa_email_response.raise_for_status()
        current_sa = sa_email_response.text
        logger.info(f"El componente update-data-op se está ejecutando con la cuenta de servicio (desde metadata): {current_sa}")
    except Exception as sa_err:
        logger.warning(f"No se pudo obtener la cuenta de servicio actual del metadata server: {sa_err}")
        logger.warning("Esto es normal si se ejecuta localmente fuera de GCP. En GCP, podría indicar un problema de red/configuración del metadata server.")

    try:
        logger.info(f"Accediendo al secreto: projects/{project_id}/secrets/{polygon_api_key_secret_name}/versions/{polygon_api_key_secret_version}")
        client = sm_client_lib.SecretManagerServiceClient()
        secret_path = client.secret_version_path(
            project_id,
            polygon_api_key_secret_name,
            polygon_api_key_secret_version
        )
        response = client.access_secret_version(request={"name": secret_path})
        polygon_api_key = response.payload.data.decode("UTF-8")
        os.environ["POLYGON_API_KEY"] = polygon_api_key
        logger.info(f"API Key de Polygon obtenida de Secret Manager ({polygon_api_key_secret_name}) y configurada en el entorno.")
    except Exception as e:
        logger.error(f"Error al obtener la API Key de Polygon de Secret Manager: {e}", exc_info=True)
        raise RuntimeError(f"Fallo al obtener la API Key de Secret Manager '{polygon_api_key_secret_name}': {e}")

    logger.info(f"Initializing data_orchestrator.py for {pair} {timeframe}...")
    message_dict = {"symbols": [pair], "timeframes": [timeframe]}
    message_str = json.dumps(message_dict)

    command = [
        "python", "data_orchestrator.py",
        "--mode", "on-demand",
        "--message", message_str,
        "--project_id", project_id
    ]
    logger.info(f"Command to execute for data_orchestrator.py: {' '.join(command)}")

    current_env = os.environ.copy()
    process = subprocess.run(command, capture_output=True, text=True, check=False, env=current_env)

    if process.returncode != 0:
        error_message = f"data_orchestrator.py failed for {pair} {timeframe}."
        stdout_output = process.stdout.strip()
        stderr_output = process.stderr.strip()
        if stdout_output: error_message += f"\nSTDOUT:\n{stdout_output}"
        if stderr_output: error_message += f"\nSTDERR:\n{stderr_output}"
        logger.error(error_message)
        raise RuntimeError(error_message)
    else:
        logger.info(f"data_orchestrator.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout.strip()}")
        if process.stderr.strip():
            logger.info(f"STDERR (aunque exit code fue 0):\n{process.stderr.strip()}")
    return f"Data update process completed for {pair}/{timeframe}."


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
    packages_to_install=["google-cloud-storage", "gcsfs", "pandas", "numpy"]
)
def prepare_opt_data_op(
    gcs_bucket_name: str,
    pair: str,
    timeframe: str,
    output_gcs_prefix: str,
    project_id: str,
) -> str:
    # TODO: Consider parameterizing the date range for data selection (e.g., "last_n_days", "start_date", "end_date")
    # This would allow more flexibility in defining the dataset for optimization.
    # Currently, prepare_opt_data.py might have a hardcoded or implicit logic for "recent" data.
    import subprocess
    from datetime import datetime
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    input_gcs_path = f"gs://{gcs_bucket_name}/data/{pair}/{timeframe}/{pair}_{timeframe}.parquet"
    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_gcs_path = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}/{pair}_{timeframe}_recent.parquet"
    logger.info(f"Initializing prepare_opt_data.py for {pair} {timeframe} (Input: {input_gcs_path}, Output: {output_gcs_path})")
    command = ["python", "prepare_opt_data.py", "--input_path", input_gcs_path, "--output_path", output_gcs_path]
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        logger.error(f"Error in prepare_opt_data.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}")
        raise RuntimeError(f"prepare_opt_data.py failed for {pair} {timeframe}")
    else:
        logger.info(f"prepare_opt_data.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}")
    return output_gcs_path


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "optuna", "scikit-learn", "joblib",
        "kfp-pipeline-spec>=0.1.16"
    ]
)
def optimize_lstm_op(
    gcs_bucket_name: str,
    features_path: str,
    pair: str,
    timeframe: str,
    n_trials: int,
    output_gcs_prefix: str,
    project_id: str,
    optimization_metrics: Output[Metrics]
) -> str:
    import subprocess
    from datetime import datetime
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    params_output_dir = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}"
    final_params_gcs_path = f"{params_output_dir}/best_params.json"

    logger.info(f"Initializing optimize_lstm.py for {pair} {timeframe} (Features: {features_path}, Params Output Dir: {params_output_dir}, Final Params File: {final_params_gcs_path}, Trials: {n_trials})")

    command = [
        "python", "optimize_lstm.py",
        "--features", features_path,
        "--pair", pair,
        "--timeframe", timeframe,
        "--output", final_params_gcs_path,
        "--n-trials", str(n_trials),
    ]
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        error_msg = f"Error in optimize_lstm.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        logger.info(f"optimize_lstm.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}")
        try:
            best_score_line = None
            for line in process.stdout.splitlines():
                if "Best is trial" in line and "with value:" in line:
                    best_score_line = line
                    break
            if best_score_line:
                parts = best_score_line.split("with value:")
                if len(parts) > 1:
                    score_str = parts[1].split(" ")[1].replace(".", "")
                    score_value_str = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', score_str.split()[0]))
                    score_value = float(score_value_str)
                    optimization_metrics.log_metric("optuna_best_trial_score", score_value)
                    logger.info(f"Optuna best trial score metric logged: {score_value}")
            else:
                logger.warning("Could not find Optuna best trial score in stdout for metrics logging.")
        except Exception as e:
            logger.warning(f"Error parsing Optuna best trial score: {e}. Metrics may not be fully logged. Full stdout was:\n{process.stdout}")
    return final_params_gcs_path


@dsl.component(
    base_image="python:3.9-slim", # Imagen para el lanzador KFP, no para el Custom Job
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"]
)
def train_lstm_vertex_ai_op(
    project_id: str,
    region: str,
    gcs_bucket_name: str,
    params_path: str,
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, # This is the base directory for versioned artifacts
    vertex_training_image_uri: str, # Esta es la URI de tu imagen runner-lstm
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
) -> str:
    import logging
    import time
    import json
    from datetime import datetime, timezone
    from google.cloud import aiplatform as gcp_aiplatform
    from google.cloud import storage as gcp_storage

    # Configurar logger
    for handler_idx in range(len(logging.root.handlers)): # Bucle seguro para eliminar
        logging.root.removeHandler(logging.root.handlers[0])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s [%(funcName)s]: %(message)s")
    logger = logging.getLogger(__name__)

    staging_gcs_path = f"gs://{gcs_bucket_name}/staging_for_custom_jobs"
    logger.info(f"Configurando staging bucket para Vertex AI Custom Job: {staging_gcs_path}")
    gcp_aiplatform.init(project=project_id, location=region, staging_bucket=staging_gcs_path)
    logger.info(f"Vertex AI SDK inicializado. Project: {project_id}, Region: {region}, Staging Bucket: {staging_gcs_path}")

    timestamp_for_job = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_display_name = f"lstm-training-{pair.lower()}-{timeframe.lower()}-{timestamp_for_job}"

    # Arguments for the train_lstm.py script
    training_script_args = [
        "--params", params_path,
        "--output-gcs-base-dir", output_gcs_prefix, # Carpeta base donde se guardan los artefactos versionados
        "--pair", pair,                             # Símbolo de mercado (ej. EURUSD)
        "--timeframe", timeframe,                   # Timeframe (ej. 15minute)
        "--project-id", project_id,                 # ID del proyecto GCP
        "--gcs-bucket-name", gcs_bucket_name        # Bucket principal de almacenamiento
    ]

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": vertex_machine_type,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": vertex_training_image_uri,
            "args": training_script_args,
            # "command" se omite para usar el ENTRYPOINT de la imagen runner-lstm
        },
    }]

    if vertex_accelerator_count > 0 and vertex_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        worker_pool_specs[0]["machine_spec"]["accelerator_type"] = vertex_accelerator_type
        worker_pool_specs[0]["machine_spec"]["accelerator_count"] = vertex_accelerator_count
        logger.info(f"Configurando Custom Job con GPU: {vertex_accelerator_count} x {vertex_accelerator_type}")
    else:
        logger.info(f"Configurando Custom Job solo con CPU.")

    # The base_output_dir for the Vertex AI CustomJob itself.
    # The train_lstm.py script will create its own versioned subdirectory inside output_gcs_prefix.
    vertex_job_base_output_dir = f"gs://{gcs_bucket_name}/vertex_ai_job_outputs/{job_display_name}"

    logger.info(f"Submitting Vertex AI Custom Job: {job_display_name}")
    logger.info(f"  Training script args: {json.dumps(training_script_args)}")
    logger.info(f"  Vertex AI Custom Job base output directory (for job metadata): {vertex_job_base_output_dir}")
    logger.info(f"  LSTM model artifacts base output directory (passed to script): {output_gcs_prefix}")


    custom_job = gcp_aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=vertex_job_base_output_dir, # For Vertex AI job artifacts
        project=project_id,
        location=region,
    )
    custom_job_launch_time_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    try:
        custom_job.run(service_account=vertex_service_account, sync=True, timeout=10800) # 3 horas timeout
        logger.info(f"Vertex AI Custom Job {custom_job.display_name} (ID: {custom_job.resource_name}) completed.")
    except Exception as e:
        logger.error(f"Vertex AI Custom Job {job_display_name} failed or timed out: {e}")
        if custom_job and custom_job.resource_name:
             logger.error(f"Job details: {custom_job.resource_name}, state: {custom_job.state}")
        raise RuntimeError(f"Vertex AI Custom Job for LSTM training failed: {e}")

    if not output_gcs_prefix.startswith("gs://"):
        raise ValueError("output_gcs_prefix debe ser una ruta GCS (gs://...)")

    # The train_lstm.py script is now responsible for creating a timestamped directory
    # inside output_gcs_prefix. We need to find that directory.
    # Example: output_gcs_prefix = gs://<bucket>/models/LSTM_v2
    # train_lstm.py creates: gs://<bucket>/models/LSTM_v2/<pair>/<timeframe>/<timestamp>/
    # We poll for this <timestamp> directory.

    prefix_parts = output_gcs_prefix.replace("gs://", "").split("/")
    expected_bucket_name = prefix_parts[0]
    # The listing prefix should be output_gcs_prefix + /<pair>/<timeframe>/
    gcs_listing_prefix = f"{'/'.join(prefix_parts[1:])}/{pair}/{timeframe}/"
    if not gcs_listing_prefix.endswith('/'): gcs_listing_prefix += '/'

    logger.info(f"Polling GCS bucket '{expected_bucket_name}' under prefix '{gcs_listing_prefix}' for model artifacts directory created by the training script (after {custom_job_launch_time_utc.isoformat()}).")

    storage_client = gcp_storage.Client(project=project_id)
    bucket = storage_client.bucket(expected_bucket_name)
    max_poll_wait_sec, poll_interval_sec = 15 * 60, 30 # 15 minutes
    start_poll_time = time.time()
    found_model_dir_gcs_path = None

    while time.time() - start_poll_time < max_poll_wait_sec:
        logger.info(f"GCS poll elapsed: {int(time.time() - start_poll_time)}s / {max_poll_wait_sec}s. Checking prefix: gs://{expected_bucket_name}/{gcs_listing_prefix}")
        blobs = bucket.list_blobs(prefix=gcs_listing_prefix, delimiter="/") # delimiter lists "subdirectories"
        candidate_dirs = []
        for page in blobs.pages: # Iterate through pages of results
            if hasattr(page, 'prefixes') and page.prefixes: # page.prefixes contains the "subdirectories"
                for dir_prefix_full_path in page.prefixes: # e.g., models/LSTM_v2/EURUSD/15minute/20230101120000/
                    # Check if model.h5 exists within this subdirectory
                    model_blob_path = f"{dir_prefix_full_path.rstrip('/')}/model.h5" # Path within the bucket
                    if bucket.blob(model_blob_path).exists():
                        # Verify the directory name is a timestamp (format %Y%m%d%H%M%S)
                        try:
                            dir_name_part = dir_prefix_full_path.rstrip('/').split('/')[-1]
                            datetime.strptime(dir_name_part, "%Y%m%d%H%M%S") # Validate format
                            candidate_dirs.append(f"gs://{expected_bucket_name}/{dir_prefix_full_path.rstrip('/')}")
                        except ValueError:
                            logger.warning(f"Directory {dir_prefix_full_path} does not seem to be a timestamped model directory. Skipping.")
                            continue
        if candidate_dirs:
            # If multiple timestamped directories are found (e.g., from previous runs or quick retries),
            # pick the latest one. The train_lstm.py script creates a new timestamped dir each time.
            found_model_dir_gcs_path = sorted(candidate_dirs, reverse=True)[0]
            logger.info(f"Found LSTM model directory in GCS: {found_model_dir_gcs_path}")
            return found_model_dir_gcs_path # Return the GCS path to the versioned model directory
        time.sleep(poll_interval_sec)

    err_msg = f"LSTM model directory (containing model.h5) not found in gs://{expected_bucket_name}/{gcs_listing_prefix} after {max_poll_wait_sec / 60:.1f} min. Vertex Job: {custom_job.display_name}"
    logger.error(err_msg)
    raise TimeoutError(err_msg)


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "joblib",
    ]
)
def prepare_rl_data_op(
    gcs_bucket_name: str,
    lstm_model_dir: str, # This is the versioned GCS path from train_lstm_vertex_ai_op
    pair: str,
    timeframe: str,
    output_gcs_prefix: str,
    project_id: str,
) -> str:
    import subprocess
    from datetime import datetime
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if not lstm_model_dir or not lstm_model_dir.startswith("gs://"):
        raise ValueError(f"prepare_rl_data_op requiere una ruta GCS válida para lstm_model_dir, se obtuvo: {lstm_model_dir}")

    # lstm_model_dir is already the versioned path like gs://<bucket>/models/LSTM_v2/<pair>/<timeframe>/<timestamp>/
    lstm_model_path = f"{lstm_model_dir.rstrip('/')}/model.h5"
    lstm_scaler_path = f"{lstm_model_dir.rstrip('/')}/scaler.pkl"
    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json" # This params.json is the one COPIED by train_lstm.py into its output

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # Create a versioned output path for RL data as well
    rl_data_output_path = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}/ppo_input_data.npz"

    logger.info(f"Initializing prepare_rl_data.py for {pair} {timeframe}. LSTM model dir: {lstm_model_dir}. Output NPZ: {rl_data_output_path}")
    command = [
        "python", "prepare_rl_data.py",
        "--model", lstm_model_path,
        "--scaler", lstm_scaler_path,
        "--params", lstm_params_path, # This should be the params.json from the LSTM model's artifact directory
        "--output", rl_data_output_path,
        "--pair", pair,
        "--timeframe", timeframe,
        "--gcs-bucket-name", gcs_bucket_name, # Pass the bucket name for the script to read raw data
    ]
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        error_msg = f"Error in prepare_rl_data.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        logger.info(f"prepare_rl_data.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}")
    return rl_data_output_path


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "stable-baselines3[extra]", "gymnasium", "pandas_ta", "scikit-learn",
        "joblib", "optuna",
    ]
)
def train_rl_op(
    gcs_bucket_name: str,
    lstm_model_dir: str, # This is the versioned GCS path from train_lstm_vertex_ai_op
    rl_data_path: str,   # This is the versioned GCS path from prepare_rl_data_op
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, # Base GCS path for RL model outputs
    tensorboard_logs_gcs_prefix: str,
    project_id: str,
) -> str: # Returns the GCS path to the trained RL model zip file
    import subprocess
    from datetime import datetime
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if not lstm_model_dir or not lstm_model_dir.startswith("gs://"):
        raise ValueError(f"train_rl_op requiere una ruta GCS válida para lstm_model_dir, se obtuvo: {lstm_model_dir}")

    # lstm_params_path should point to the params.json COPIED by train_lstm.py into its output directory
    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json"
    tensorboard_log_gcs_path = f"{tensorboard_logs_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    logger.info(f"Initializing train_rl.py for {pair} {timeframe}. LSTM params (from LSTM artifact dir): {lstm_params_path}, RL Data: {rl_data_path}, Output Base: {output_gcs_prefix}, TensorBoard: {tensorboard_log_gcs_path}")
    command = [
        "python", "train_rl.py",
        "--params", lstm_params_path, # This is the params.json from the LSTM model artifact directory
        "--rl-data", rl_data_path,
        "--output-bucket", gcs_bucket_name, # Bucket for train_rl.py to write to
        "--pair", pair,
        "--timeframe", timeframe,
        "--output-model-base-gcs-path", output_gcs_prefix, # train_rl.py will create versioned subdirs here
        "--tensorboard-log-dir", tensorboard_log_gcs_path,
    ]
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        error_msg = f"Error in train_rl.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        logger.info(f"train_rl.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}")
        # train_rl.py should print the GCS path of the saved model.
        # Example expected output line: "RL model saved to gs://<bucket>/models/RL_v2/<pair>/<timeframe>/<timestamp>/ppo_filter_model.zip"
        rl_model_upload_line = next((l for l in process.stdout.splitlines() if "RL model saved to gs://" in l or "Subido ppo_filter_model.zip a" in l), None)
        if rl_model_upload_line:
            if "RL model saved to " in rl_model_upload_line:
                 uploaded_rl_path = rl_model_upload_line.split("RL model saved to ")[1].strip()
            elif "Subido ppo_filter_model.zip a " in rl_model_upload_line: # Legacy or alternative phrasing
                 uploaded_rl_path = rl_model_upload_line.split("Subido ppo_filter_model.zip a ")[1].strip()
            else: # Fallback if the exact phrase isn't matched but a gs:// path is there
                logger.warning("Could not precisely determine RL model path from stdout using known patterns. Looking for any 'gs://' path ending in .zip.")
                gs_paths_in_stdout = [word for word in process.stdout.split() if word.startswith("gs://") and word.endswith(".zip")]
                if gs_paths_in_stdout:
                    uploaded_rl_path = gs_paths_in_stdout[-1] # Assume the last one is the relevant one
                    logger.info(f"Guessed RL model path from stdout: {uploaded_rl_path}")
                else:
                    raise RuntimeError("Failed to determine RL model output path from train_rl.py stdout. No 'gs://...zip' path found.")
            logger.info(f"RL model uploaded to: {uploaded_rl_path}")
            return uploaded_rl_path.rstrip('/') # Return the full GCS path to the .zip file
        else:
            raise RuntimeError("Failed to determine RL model output path from train_rl.py stdout. Ensure the script prints the GCS path like 'RL model saved to gs://...'.")


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "stable-baselines3", "gymnasium", "pandas_ta", "scikit-learn",
        "joblib", "optuna",
        "kfp-pipeline-spec>=0.1.16"
    ]
)
def backtest_op(
    lstm_model_dir: str, # Versioned GCS path from train_lstm_vertex_ai_op
    rl_model_path: str,  # GCS path to the RL model zip file from train_rl_op
    features_path: str,  # GCS path to the unseen backtest data
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, # Base GCS path for backtest outputs
    project_id: str,
    backtest_metrics: Output[Metrics]
) -> str: # Returns GCS path to the directory containing backtest results (including metrics.json)
    import subprocess
    from datetime import datetime
    import logging
    import json as py_json
    from google.cloud import storage as gcp_storage
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if not lstm_model_dir.startswith("gs://") or not rl_model_path.startswith("gs://"):
        raise ValueError("backtest_op requiere rutas GCS válidas para los modelos.")

    # Construct paths to individual files within the versioned LSTM model directory
    lstm_model_file_path = f"{lstm_model_dir.rstrip('/')}/model.h5"
    lstm_scaler_path = f"{lstm_model_dir.rstrip('/')}/scaler.pkl"
    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json" # The one copied by train_lstm.py

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # Backtest script will save its outputs (including metrics.json) into this versioned directory
    backtest_output_gcs_dir = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}/"
    expected_metrics_file_gcs_path = f"{backtest_output_gcs_dir.rstrip('/')}/metrics.json"

    logger.info(f"Initializing backtest.py for {pair} {timeframe}. Output dir: {backtest_output_gcs_dir}")
    command = [
        "python", "backtest.py",
        "--pair", pair,
        "--timeframe", timeframe,
        "--lstm-model-path", lstm_model_file_path,
        "--lstm-scaler-path", lstm_scaler_path,
        "--lstm-params-path", lstm_params_path, # Params from LSTM artifact dir
        "--rl-model-path", rl_model_path,       # Path to the RL .zip model
        "--features-path", features_path,       # Unseen data for backtesting
        "--output-dir", backtest_output_gcs_dir, # Script will save all outputs here
    ]
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        error_msg = f"Error in backtest.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        logger.info(f"backtest.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}")
        try:
            storage_client = gcp_storage.Client(project=project_id)
            path_parts = expected_metrics_file_gcs_path.replace("gs://", "").split("/", 1)
            bucket_name_str = path_parts[0]
            blob_path_str = path_parts[1]

            bucket = storage_client.bucket(bucket_name_str)
            blob = bucket.blob(blob_path_str)

            if blob.exists():
                logger.info(f"Attempting to download metrics from: {expected_metrics_file_gcs_path}")
                metrics_content = blob.download_as_string()
                metrics_data = py_json.loads(metrics_content)
                logger.info(f"Metrics data loaded: {metrics_data}")

                metrics_to_log = metrics_data.get("filtered", metrics_data)

                for key, value in metrics_to_log.items():
                    if isinstance(value, (int, float)):
                        backtest_metrics.log_metric(key, value)
                    else:
                        try:
                            float_value = float(value)
                            backtest_metrics.log_metric(key, float_value)
                        except (ValueError, TypeError):
                            logger.warning(f"Cannot log metric '{key}' with value '{value}' (type: {type(value)}) as float. Skipping.")
                logger.info(f"Logged metrics from {expected_metrics_file_gcs_path} (section 'filtered' or root) to KFP Metrics.")
            else:
                logger.warning(f"Metrics file {expected_metrics_file_gcs_path} not found for KFP logging. STDOUT from backtest.py might contain clues:\n{process.stdout}")
        except Exception as e_metrics:
            logger.error(f"Failed to log metrics from {expected_metrics_file_gcs_path} to KFP: {e_metrics}", exc_info=True)
    return backtest_output_gcs_dir.rstrip('/') # Return the GCS path to the directory


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
    packages_to_install=["google-cloud-storage", "gcsfs", "pandas", "numpy"]
)
def decide_promotion_op(
    new_backtest_metrics_dir: str, # GCS path to directory from backtest_op
    new_lstm_artifacts_dir: str, # GCS path to versioned LSTM model dir from train_lstm_vertex_ai_op
    new_rl_model_path: str,      # GCS path to RL model .zip file from train_rl_op
    gcs_bucket_name: str,
    pair: str,
    timeframe: str,
    project_id: str,
    current_production_metrics_path: str, # GCS path to current prod metrics.json
    production_base_dir: str, # Base GCS path for production models (e.g., gs://<bucket>/models/production_v2)
) -> bool: # Returns True if model promoted, False otherwise
    import subprocess
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # new_backtest_metrics_dir is the directory containing metrics.json
    new_metrics_file_path = f"{new_backtest_metrics_dir.rstrip('/')}/metrics.json"

    if not new_lstm_artifacts_dir.startswith("gs://") or \
       not new_rl_model_path.startswith("gs://") or \
       not new_metrics_file_path.startswith("gs://"): # Check the derived metrics file path
        raise ValueError("decide_promotion_op requiere rutas GCS válidas para artefactos y métricas.")

    # Target directory for promotion, e.g., gs://<bucket>/models/production_v2/<pair>/<timeframe>/
    production_pair_timeframe_dir = f"{production_base_dir.rstrip('/')}/{pair}/{timeframe}"

    logger.info(f"Initializing model promotion decision for {pair} {timeframe}.")
    logger.info(f"  New metrics file: {new_metrics_file_path}")
    logger.info(f"  Current prod metrics: {current_production_metrics_path}")
    logger.info(f"  New LSTM dir (source for promotion): {new_lstm_artifacts_dir}")
    logger.info(f"  New RL model (source for promotion): {new_rl_model_path}")
    logger.info(f"  Production target dir (destination for promotion): {production_pair_timeframe_dir}")

    command = [
        "python", "model_promotion_decision.py",
        "--new-metrics-path", new_metrics_file_path,
        "--current-production-metrics-path", current_production_metrics_path,
        "--new-lstm-artifacts-dir", new_lstm_artifacts_dir, # Pass the whole dir
        "--new-rl-model-path", new_rl_model_path,         # Pass the .zip model path
        "--production-pair-timeframe-dir", production_pair_timeframe_dir, # Target dir for copying
        "--pair", pair,
        "--timeframe", timeframe,
        # --project-id and --gcs-bucket-name might be needed if model_promotion_decision.py interacts with GCS directly
        # Add them here if required by the script. For now, assuming it primarily uses paths.
    ]
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    model_promoted_flag = False
    if process.returncode == 0:
        logger.info(f"model_promotion_decision.py completed.\nSTDOUT: {process.stdout}")
        # Check stdout for a clear signal of promotion.
        if "PROMOVIDO a producción." in process.stdout or "Model promoted to production" in process.stdout.lower():
            model_promoted_flag = True
            logger.info("Model promotion confirmed based on script output.")
        else:
            logger.info("Model not promoted based on script output.")
    else:
        # If the script fails, assume no promotion.
        logger.error(f"Error in model_promotion_decision.py. Model will not be promoted.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}")

    logger.info(f"Model promoted: {model_promoted_flag}")
    return model_promoted_flag


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
    packages_to_install=["google-cloud-pubsub"],
)
def notify_pipeline_status_op(
    project_id: str,
    pair: str,
    timeframe: str,
    pipeline_run_status: str, # This will be set by KFP ExitHandler or dsl.If conditions
    pipeline_job_id: str,
    pipeline_job_link: str,
    model_promoted: bool = False, # Default to False, set to True if promotion occurs
) -> bool:
    from google.cloud import pubsub_v1
    import logging
    import json as py_json
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    SUCCESS_TOPIC_ID = "data-ingestion-success" # Or a more generic pipeline success topic
    FAILURE_TOPIC_ID = "data-ingestion-failures" # Or a more generic pipeline failure topic

    try:
        publisher = pubsub_v1.PublisherClient()
        ts_utc_iso = datetime.now(timezone.utc).isoformat()

        # Normalize status for KFP v2 values if needed
        status_summary = pipeline_run_status.upper()
        # KFP v2 uses "SUCCEEDED", "FAILED", "CANCELLED"
        # The ExitHandler might pass dsl.PIPELINE_STATUS_PLACEHOLDER which resolves to these.
        if status_summary not in ["SUCCEEDED", "FAILED", "CANCELLED", "UNKNOWN_STATUS"]: # UNKNOWN_STATUS is my placeholder for ExitHandler
            logger.warning(f"Pipeline status '{pipeline_run_status}' is not one of the expected canonical KFP v2 statuses (SUCCEEDED, FAILED, CANCELLED) or UNKNOWN_STATUS. Will attempt to normalize.")
            if "fail" in pipeline_run_status.lower() or "error" in pipeline_run_status.lower():
                status_summary = "FAILED"
            elif "success" in pipeline_run_status.lower() or "succeeded" in pipeline_run_status.lower():
                status_summary = "SUCCEEDED"
            else: # If it's something else from a direct call, treat as unknown/failure for notification
                status_summary = "UNKNOWN_STATUS" # This will route to failure topic

        details = {
            "pipeline_name": "algo-trading-mlops-gcp-pipeline-v2",
            "pipeline_job_id": pipeline_job_id,
            "pipeline_job_link": pipeline_job_link,
            "pair": pair,
            "timeframe": timeframe,
            "run_status": status_summary, # Use the (potentially normalized) status
            "model_promoted_status": model_promoted if status_summary == "SUCCEEDED" else False, # Only relevant on success
            "timestamp_utc": ts_utc_iso,
        }

        target_topic_id = SUCCESS_TOPIC_ID if status_summary == "SUCCEEDED" else FAILURE_TOPIC_ID
        if status_summary == "UNKNOWN_STATUS": # UNKNOWN_STATUS from ExitHandler default should go to failure
             target_topic_id = FAILURE_TOPIC_ID

        summary_message = f"MLOps Pipeline Notification: {details['pipeline_name']} for {pair}/{timeframe} finished with status: {status_summary}."
        if status_summary == "SUCCEEDED":
            summary_message += f" Model Promoted: {model_promoted}."

        msg_data = {"summary": summary_message, "details": details}
        msg_bytes = py_json.dumps(msg_data, indent=2).encode("utf-8")
        topic_path = publisher.topic_path(project_id, target_topic_id)

        future = publisher.publish(topic_path, msg_bytes)
        message_id = future.result(timeout=60) # Wait for publish to complete
        logger.info(f"Notification sent to Pub/Sub topic '{topic_path}' with message ID: {message_id}.")
        return True
    except Exception as e:
        logger.error(f"Error publishing notification to Pub/Sub: {e}", exc_info=True)
        return False # Indicate failure to publish


# --- Pipeline Definition ---
@dsl.pipeline(
    name="algo-trading-mlops-gcp-pipeline-v2",
    description="KFP v2 Pipeline for training and deploying algorithmic trading models.",
    pipeline_root=PIPELINE_ROOT,
)
def training_pipeline(
    pair: str = DEFAULT_PAIR,
    timeframe: str = DEFAULT_TIMEFRAME,
    n_trials: int = DEFAULT_N_TRIALS,
    backtest_features_path: str = f"gs://{GCS_BUCKET_NAME}/backtest_data/{DEFAULT_PAIR}_{DEFAULT_TIMEFRAME}_unseen.parquet",
    vertex_lstm_training_image: str = VERTEX_LSTM_TRAINER_IMAGE_URI,
    vertex_lstm_machine_type: str = DEFAULT_VERTEX_LSTM_MACHINE_TYPE,
    vertex_lstm_accelerator_type: str = DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE,
    vertex_lstm_accelerator_count: int = DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT,
    vertex_lstm_service_account: str = DEFAULT_VERTEX_LSTM_SERVICE_ACCOUNT,
    kfp_opt_accelerator_type: str = KFP_COMPONENT_ACCELERATOR_TYPE, # Not used by optimize_lstm_op currently
    kfp_opt_accelerator_count: int = KFP_COMPONENT_ACCELERATOR_COUNT, # Not used by optimize_lstm_op currently
    polygon_api_key_secret_name_param: str = "polygon-api-key",
    polygon_api_key_secret_version_param: str = "latest",
):
    pipeline_job_id_val = dsl.PIPELINE_JOB_ID_PLACEHOLDER
    # pipeline_job_name_val = dsl.PIPELINE_JOB_NAME_PLACEHOLDER # Also available
    pipeline_ui_link_val = f"https://console.cloud.google.com/vertex-ai/pipelines/runs/{pipeline_job_id_val}?project={PROJECT_ID}"
    # For KFP v2, dsl.PIPELINE_STATUS_PLACEHOLDER holds the final status (SUCCEEDED, FAILED, CANCELLED)
    
        # Generic Exit Handler: always runs, sends FAILURE unless explicitly overridden by success notifications
    exit_notify_task = notify_pipeline_status_op(
        project_id=PROJECT_ID,
        pair=pair,
        timeframe=timeframe,
        pipeline_run_status="UNKNOWN_STATUS",  # valor de reserva si no se conoce
        pipeline_job_id=pipeline_job_id_val,
        pipeline_job_link=pipeline_ui_link_val,
        model_promoted=False
    ).set_display_name("Notify_Pipeline_Final_Status_Exit_Handler")
    # Ensure this runs regardless of upstream failures if possible (though KFP handles this)
    exit_notify_task.set_cpu_limit("1").set_memory_limit("1G")


    with dsl.ExitHandler(exit_notify_task):
        update_data_task = update_data_op(
            pair=pair,
            timeframe=timeframe,
            gcs_bucket_name=GCS_BUCKET_NAME,
            project_id=PROJECT_ID,
            polygon_api_key_secret_name=polygon_api_key_secret_name_param,
            polygon_api_key_secret_version=polygon_api_key_secret_version_param
        )
        update_data_task.set_display_name("Update_Market_Data")
        update_data_task.set_cpu_limit("2").set_memory_limit("4G")

        prepare_opt_data_task = prepare_opt_data_op(
            gcs_bucket_name=GCS_BUCKET_NAME,
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/data_filtered_for_opt_v2", # Output for this step
            project_id=PROJECT_ID
        ).after(update_data_task)
        prepare_opt_data_task.set_display_name("Prepare_Optimization_Data")
        prepare_opt_data_task.set_cpu_limit("4").set_memory_limit("15G")

        optimize_lstm_task = optimize_lstm_op(
            gcs_bucket_name=GCS_BUCKET_NAME,
            features_path=prepare_opt_data_task.output, # Output from prepare_opt_data_op
            pair=pair,
            timeframe=timeframe,
            n_trials=n_trials,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/params/LSTM_v2", # Output for this step (best_params.json location)
            project_id=PROJECT_ID
        )
        optimize_lstm_task.set_display_name("Optimize_LSTM_Hyperparameters")
        optimize_lstm_task.set_cpu_limit("8").set_memory_limit("30G")
        # GPUs for KFP components (not Vertex Custom Job) would be like:
        # if kfp_opt_accelerator_count > 0 and kfp_opt_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        #    optimize_lstm_task.set_gpu_limit(kfp_opt_accelerator_count).add_node_selector_constraint('cloud.google.com/gke-accelerator', kfp_opt_accelerator_type)


        # train_lstm_vertex_ai_op's 'output_gcs_prefix' is where IT will save its versioned model artifacts
        # This is gs://<GCS_BUCKET_NAME>/models/LSTM_v2
        # The component itself will then return the full GCS path to the created versioned sub-directory.
        train_lstm_task = train_lstm_vertex_ai_op(
            project_id=PROJECT_ID,
            region=REGION,
            gcs_bucket_name=GCS_BUCKET_NAME,
            params_path=optimize_lstm_task.outputs['Output'], # GCS path to best_params.json
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/models/LSTM_v2", # Base dir for LSTM model artifacts
            vertex_training_image_uri=vertex_lstm_training_image,
            vertex_machine_type=vertex_lstm_machine_type,
            vertex_accelerator_type=vertex_lstm_accelerator_type,
            vertex_accelerator_count=vertex_lstm_accelerator_count,
            vertex_service_account=vertex_lstm_service_account
        )
        train_lstm_task.set_display_name("Train_LSTM_Model_Vertex_AI")
        train_lstm_task.set_cpu_limit("1").set_memory_limit("2G") # For KFP launcher, not the Vertex job

        # train_lstm_task.output is the GCS path to the versioned LSTM model directory,
        # e.g., gs://<bucket>/models/LSTM_v2/<pair>/<timeframe>/<timestamp>/
        prepare_rl_data_task = prepare_rl_data_op(
            gcs_bucket_name=GCS_BUCKET_NAME,
            lstm_model_dir=train_lstm_task.output, # Pass the versioned LSTM model dir
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/data_rl_inputs_v2", # Base dir for RL data NPZ files
            project_id=PROJECT_ID
        )
        prepare_rl_data_task.set_display_name("Prepare_RL_Data")
        prepare_rl_data_task.set_cpu_limit("8").set_memory_limit("30G")

        # train_rl_task also takes lstm_model_dir (for params.json) and rl_data_path (for NPZ)
        # Its output_gcs_prefix is where IT will save its versioned RL model artifacts.
        train_rl_task = train_rl_op(
            gcs_bucket_name=GCS_BUCKET_NAME,
            lstm_model_dir=train_lstm_task.output, # Pass the versioned LSTM model dir
            rl_data_path=prepare_rl_data_task.output, # Pass the GCS path to the RL data NPZ file
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/models/RL_v2", # Base dir for RL model .zip files
            tensorboard_logs_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/tensorboard_logs_v2",
            project_id=PROJECT_ID
        )
        train_rl_task.set_display_name("Train_PPO_Agent")
        train_rl_task.set_cpu_limit("16").set_memory_limit("60G") # High resources for RL training

        # backtest_task uses the versioned LSTM model dir and the specific RL model .zip file
        # Its output_gcs_prefix is where IT will save its versioned backtest results.
        backtest_task = backtest_op(
            lstm_model_dir=train_lstm_task.output,     # Versioned LSTM model dir
            rl_model_path=train_rl_task.output,       # Specific RL model .zip GCS path
            features_path=backtest_features_path,     # Unseen data for backtest
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/backtest_results_v2", # Base dir for backtest outputs
            project_id=PROJECT_ID
        )
        backtest_task.set_display_name("Execute_Full_Backtesting")
        backtest_task.set_cpu_limit("8").set_memory_limit("30G")

        # decide_task uses the output directory from backtest (for metrics.json),
        # the versioned LSTM dir, and the specific RL model .zip file for potential promotion.
        decide_task = decide_promotion_op(
            new_backtest_metrics_dir=backtest_task.outputs['Output'], # Dir containing metrics.json
            new_lstm_artifacts_dir=train_lstm_task.output,    # Versioned LSTM model dir to promote
            new_rl_model_path=train_rl_task.output,          # Specific RL model .zip to promote
            gcs_bucket_name=GCS_BUCKET_NAME,
            pair=pair,
            timeframe=timeframe,
            project_id=PROJECT_ID,
            current_production_metrics_path=f"gs://{GCS_BUCKET_NAME}/models/production_v2/{pair}/{timeframe}/metrics_production.json",
            production_base_dir=f"gs://{GCS_BUCKET_NAME}/models/production_v2" # Base dir for "production" models
        )
        decide_task.set_display_name("Decide_Model_Promotion")
        decide_task.set_cpu_limit("2").set_memory_limit("4G")

        # Conditional notifications based on promotion decision (these run if pipeline reaches this far)
        # These specific notifications effectively "override" the generic ExitHandler's assumption of failure
        # if the pipeline is successful.
        with dsl.If(decide_task.output == True, name="If_Model_Promoted"):
            # This task's `pipeline_run_status` is explicitly "SUCCEEDED"
            # This will replace the ExitHandler's message IF the pipeline reaches here successfully.
            notify_promoted_task = notify_pipeline_status_op(
                project_id=PROJECT_ID,
                pair=pair,
                timeframe=timeframe,
                pipeline_run_status="SUCCEEDED", # Explicitly SUCCEEDED
                model_promoted=True,
                pipeline_job_id=pipeline_job_id_val,
                pipeline_job_link=pipeline_ui_link_val,
            )
            notify_promoted_task.set_display_name("Notify_Pipeline_Success_Promoted")
            # This task should not run if the main pipeline fails before this point.
            # The ExitHandler would have already caught that.
            # We need to ensure the ExitHandler's notification isn't sent if *this* success notification is.
            # KFP's ExitHandler runs *after* the main pipeline finishes (successfully or not).
            # The `pipeline_status_val` in the exit handler will correctly reflect SUCCEEDED if we reach here.
            # The `model_promoted` flag in the exit_notify_task needs to be updated.
            # This can be done by making the exit_notify_task depend on the output of decide_task,
            # but ExitHandlers cannot easily consume outputs from within the pipeline body directly for their parameters.
            # A simpler way: the ExitHandler sends its message based on the *final* pipeline status.
            # If the pipeline succeeds AND this 'If_Model_Promoted' branch runs, then 'model_promoted' is true.
            # We can have the exit handler take `decide_task.output` if it's successful.
            # However, KFP v2 ExitHandler parameters are evaluated at pipeline compile time or must be placeholders.
            # The current setup relies on the `pipeline_status_val` passed to the ExitHandler.
            # The `model_promoted` in the exit handler's call is set to False initially.
            # To solve this, the `exit_notify_task` can be redefined or updated after `decide_task`.
            # This is tricky with KFP SDK's ExitHandler.
            # A more robust way is for the notify_pipeline_status_op to *optionally* take a promotion status.
            # And the ExitHandler's `model_promoted` is always what `decide_task.output` is IF the pipeline succeeds.
            # Let's refine the ExitHandler setup.

            # The model_promoted status in the ExitHandler will be updated by a final task.
            # We pass `decide_task.output` to the exit handler's notification if the pipeline is successful.
            # This is complex because exit handler args are usually static or placeholders.
            # The two dsl.If blocks below for notification are a clearer way to send the final correct status.
            # The global ExitHandler then primarily serves for actual *failures* before these dsl.If blocks.

        with dsl.If(decide_task.output == False, name="If_Model_Not_Promoted"):
            notify_not_promoted_task = notify_pipeline_status_op(
                project_id=PROJECT_ID,
                pair=pair,
                timeframe=timeframe,
                pipeline_run_status="SUCCEEDED", # Pipeline succeeded, model just not promoted
                model_promoted=False,
                pipeline_job_id=pipeline_job_id_val,
                pipeline_job_link=pipeline_ui_link_val,
            )
            notify_not_promoted_task.set_display_name("Notify_Pipeline_Success_Not_Promoted")

    # Re-define the exit_notify_task outside the main try/except, or ensure it gets the right model_promoted value.
    # The issue is that the exit_notify_task in the ExitHandler is defined with model_promoted=False.
    # If the pipeline succeeds and decide_task.output is True, we want the ExitHandler's notification
    # (if it's the one sending the "SUCCEEDED" message) to reflect model_promoted=True.

    # KFP v2 behavior: The ExitHandler's `pipeline_status_val` will be "SUCCEEDED" if the main graph completes.
    # The separate `dsl.If` blocks will send their specific notifications.
    # The ExitHandler will ALSO send a notification. To avoid duplicate "SUCCEEDED" notifications,
    # one strategy is to make the ExitHandler's success notification conditional or have it fetch the promotion status.
    # For now, the `dsl.If` blocks provide more specific final notifications for success cases.
    # The ExitHandler is the catch-all for true pipeline execution *failures*.

# --- Pipeline Compilation and Optional Direct Execution ---
if __name__ == "__main__":
    pipeline_filename = "algo_trading_mlops_gcp_pipeline_v2.json"
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=pipeline_filename
    )
    print(f"✅ KFP Pipeline (compatible con v2) compilado a {pipeline_filename}")

    SUBMIT_TO_VERTEX_AI = os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() != "false"

    if SUBMIT_TO_VERTEX_AI:
        print(f"\n🚀 Iniciando la sumisión y ejecución directa de la pipeline '{pipeline_filename}' en Vertex AI...")

        aiplatform_sdk.init(project=PROJECT_ID, location=REGION)
        print(f" Vertex AI SDK inicializado para Project: {PROJECT_ID}, Region: {REGION}")

        run_pair = os.getenv("PIPELINE_PAIR", DEFAULT_PAIR)
        run_timeframe = os.getenv("PIPELINE_TIMEFRAME", DEFAULT_TIMEFRAME)
        run_n_trials = int(os.getenv("PIPELINE_N_TRIALS", str(DEFAULT_N_TRIALS)))
        run_backtest_features_path = os.getenv(
            "PIPELINE_BACKTEST_FEATURES_PATH",
            f"gs://{GCS_BUCKET_NAME}/backtest_data/{run_pair}_{run_timeframe}_unseen.parquet"
        )
        run_polygon_secret_name = os.getenv("PIPELINE_POLYGON_SECRET_NAME", "polygon-api-key")

        pipeline_parameter_values = {
            "pair": run_pair,
            "timeframe": run_timeframe,
            "n_trials": run_n_trials,
            "backtest_features_path": run_backtest_features_path,
            "vertex_lstm_training_image": VERTEX_LSTM_TRAINER_IMAGE_URI,
            "vertex_lstm_machine_type": DEFAULT_VERTEX_LSTM_MACHINE_TYPE,
            "vertex_lstm_accelerator_type": DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE,
            "vertex_lstm_accelerator_count": DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT,
            "vertex_lstm_service_account": DEFAULT_VERTEX_LSTM_SERVICE_ACCOUNT,
            "kfp_opt_accelerator_type": KFP_COMPONENT_ACCELERATOR_TYPE, # Note: still passed but not used in op
            "kfp_opt_accelerator_count": KFP_COMPONENT_ACCELERATOR_COUNT, # Note: still passed but not used in op
            "polygon_api_key_secret_name_param": run_polygon_secret_name,
            "polygon_api_key_secret_version_param": "latest",
        }

        job_display_name = f"algo-trading-v2-{run_pair}-{run_timeframe}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        print(f" Display Name para PipelineJob: {job_display_name}")
        print(f" Template Path (archivo JSON compilado): {pipeline_filename}")
        print(f" Pipeline Root: {PIPELINE_ROOT}")
        print(" Pipeline Parameter Values:")
        print(json.dumps(pipeline_parameter_values, indent=2))

        job = aiplatform_sdk.PipelineJob(
            display_name=job_display_name,
            template_path=pipeline_filename,
            pipeline_root=PIPELINE_ROOT,
            location=REGION,
            project=PROJECT_ID, # Explicitly set project for PipelineJob
            enable_caching=False, # Recommended to disable caching for development/testing iterations
            parameter_values=pipeline_parameter_values,
        )

        job_resource_name = None
        try:
            print(f"\nEnviando PipelineJob '{job_display_name}' y esperando su finalización (ejecución síncrona)...")
            # For KFP v2, job.run() is synchronous. For async, use job.submit().
            job.run() # Use run() for synchronous execution as in the original script
            # If job.run() is used, it will raise an exception on failure.
            # If job.submit() were used, you'd poll job.state.

            job_resource_name = job.resource_name # Get after job.run() completes or if an error occurs after submission

            print(f"\n✅ PipelineJob completado.")
            print(f" Job Resource Name: {job.resource_name}")
            if hasattr(job, 'state') and job.state: # Should be PipelineState.PIPELINE_STATE_SUCCEEDED
                 print(f" Estado final del Job: {job.state}")
            # job.error might not be populated if job.run() succeeds. It's more for job.submit() polling.
            # if job.error:
            #    print(f" Job Error (si existe): {job.error}")

        except Exception as e:
            print(f"\n❌ Error durante la sumisión o ejecución del PipelineJob:")
            print(f"  Tipo de Excepción: {type(e).__name__}")
            print(f"  Mensaje: {str(e)}")
            # If the error is from google.api_core.exceptions.GoogleAPICallError, it might contain more details.
            if hasattr(e, 'errors'): print(f"  API Errors: {e.errors}")
            if hasattr(e, 'details'): print(f"  API Details: {e.details}")

            print("  Traceback:")
            traceback.print_exc()

            if job and hasattr(job, 'resource_name') and job.resource_name:
                 job_resource_name = job.resource_name # Ensure it's captured if available
                 print(f"  Job Resource Name (si se obtuvo): {job_resource_name}")
                 try:
                     # Refresh the job object to get the latest state, especially if run() failed mid-way
                     job.refresh() # Fetches the latest state from the API
                     current_job_state = job.state if hasattr(job, 'state') else "No disponible"
                     current_job_error = job.error if hasattr(job, 'error') and job.error else "No disponible" # job.error is a google.rpc.Status
                     print(f"  Estado actual del Job (refrescado): {current_job_state}")
                     if current_job_error != "No disponible":
                         print(f"  Error del Job (refrescado): {current_job_error.message if hasattr(current_job_error, 'message') else current_job_error}")
                 except Exception as refresh_err:
                     print(f"  No se pudo refrescar el estado del job: {refresh_err}")
            else:
                print("  No se pudo obtener el Job Resource Name (falló la sumisión inicial o la creación del objeto Job).")
        finally:
            if job_resource_name: # job_resource_name might be set even if job.run() fails
                job_id_for_link = job_resource_name.split('/')[-1]
                pipeline_console_link = f"https://console.cloud.google.com/vertex-ai/pipelines/runs/{job_id_for_link}?project={PROJECT_ID}&region={REGION}"
                print(f"\n🔗 Visualizar en Vertex AI Pipelines: {pipeline_console_link}")
            else:
                # This case might happen if aiplatform_sdk.PipelineJob() itself fails.
                print("\nℹ️ No se generó un enlace de ejecución ya que el PipelineJob no se pudo someter o falló muy temprano.")
    else:
        print(f"Pipeline compilado a {pipeline_filename}. Para ejecutar, sube este archivo a Vertex AI Pipelines o establece SUBMIT_PIPELINE_TO_VERTEX=true.")