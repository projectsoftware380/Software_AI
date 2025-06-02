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
PROJECT_ID = "trading-ai-460823" #
REGION = "us-central1" #

GCS_BUCKET_NAME = "trading-ai-models-460823" #
PIPELINE_ROOT = f"gs://{GCS_BUCKET_NAME}/pipeline_root_v2" #

# --- Docker Image URIs ---
KFP_COMPONENTS_IMAGE_URI = (
    f"europe-west1-docker.pkg.dev/{PROJECT_ID}/data-ingestion-repo/data-ingestion-agent:latest"
) #
VERTEX_LSTM_TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/trading-images/trainer-cpu:latest"
) #

# --- Machine Types and GPUs for Vertex AI ---
DEFAULT_VERTEX_LSTM_MACHINE_TYPE = "n1-standard-4" #
DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED" #
DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT = 0 #

KFP_COMPONENT_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED" #
KFP_COMPONENT_ACCELERATOR_COUNT = 0 #

# --- Service Account for Vertex AI Custom Jobs ---
DEFAULT_VERTEX_LSTM_SERVICE_ACCOUNT = "data-ingestion-agent@trading-ai-460823.iam.gserviceaccount.com"


# --- Default Pipeline Hyperparameters ---
DEFAULT_N_TRIALS = 2 #
DEFAULT_PAIR = "EURUSD" #
DEFAULT_TIMEFRAME = "15minute" #


# --- Component Definitions ---
@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, #
    packages_to_install=[
        "google-cloud-storage",
        "google-cloud-pubsub",
        "gcsfs",
        "pandas",
        "google-cloud-secret-manager",
        "requests",
    ] #
)
def update_data_op(
    pair: str,
    timeframe: str,
    gcs_bucket_name: str,
    project_id: str,
    polygon_api_key_secret_name: str,
    polygon_api_key_secret_version: str = "latest",
) -> str: #
    import subprocess
    import logging
    import json
    import os
    import requests
    from google.cloud import secretmanager as sm_client_lib

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s [%(funcName)s]: %(message)s") #
    logger = logging.getLogger(__name__)

    try:
        metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email" #
        headers = {"Metadata-Flavor": "Google"} #
        sa_email_response = requests.get(metadata_url, headers=headers, timeout=5) #
        sa_email_response.raise_for_status() #
        current_sa = sa_email_response.text #
        logger.info(f"El componente update-data-op se está ejecutando con la cuenta de servicio (desde metadata): {current_sa}") #
    except Exception as sa_err:
        logger.warning(f"No se pudo obtener la cuenta de servicio actual del metadata server: {sa_err}") #
        logger.warning("Esto es normal si se ejecuta localmente fuera de GCP. En GCP, podría indicar un problema de red/configuración del metadata server.") #

    try:
        logger.info(f"Accediendo al secreto: projects/{project_id}/secrets/{polygon_api_key_secret_name}/versions/{polygon_api_key_secret_version}") #
        client = sm_client_lib.SecretManagerServiceClient() #
        secret_path = client.secret_version_path(
            project_id,
            polygon_api_key_secret_name,
            polygon_api_key_secret_version
        ) #
        response = client.access_secret_version(request={"name": secret_path}) #
        polygon_api_key = response.payload.data.decode("UTF-8") #

        os.environ["POLYGON_API_KEY"] = polygon_api_key #
        logger.info(f"API Key de Polygon obtenida de Secret Manager ({polygon_api_key_secret_name}) y configurada en el entorno.") #

    except Exception as e:
        logger.error(f"Error al obtener la API Key de Polygon de Secret Manager: {e}", exc_info=True) #
        raise RuntimeError(f"Fallo al obtener la API Key de Secret Manager '{polygon_api_key_secret_name}': {e}") #

    logger.info(f"Initializing data_orchestrator.py for {pair} {timeframe}...") #
    message_dict = {"symbols": [pair], "timeframes": [timeframe]} #
    message_str = json.dumps(message_dict) #

    command = [
    "python", "data_orchestrator.py",
    "--mode", "on-demand",
    "--message", message_str,
    "--project_id", project_id
] #

    logger.info(f"Command to execute for data_orchestrator.py: {' '.join(command)}") #

    current_env = os.environ.copy() #
    process = subprocess.run(command, capture_output=True, text=True, check=False, env=current_env) #

    if process.returncode != 0:
        error_message = f"data_orchestrator.py failed for {pair} {timeframe}." #
        stdout_output = process.stdout.strip() #
        stderr_output = process.stderr.strip() #
        if stdout_output:
            error_message += f"\nSTDOUT:\n{stdout_output}" #
        if stderr_output:
            error_message += f"\nSTDERR:\n{stderr_output}" #

        logger.error(error_message) #
        raise RuntimeError(error_message) #
    else:
        logger.info(f"data_orchestrator.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout.strip()}") #
        if process.stderr.strip():
            logger.info(f"STDERR (aunque exit code fue 0):\n{process.stderr.strip()}") #
    return f"Data update process completed for {pair}/{timeframe}." #


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, #
    packages_to_install=["google-cloud-storage", "gcsfs", "pandas", "numpy"] #
)
def prepare_opt_data_op(
    gcs_bucket_name: str,
    pair: str,
    timeframe: str,
    output_gcs_prefix: str,
    project_id: str,
) -> str: #
    import subprocess
    from datetime import datetime
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") #
    logger = logging.getLogger(__name__)
    input_gcs_path = f"gs://{gcs_bucket_name}/data/{pair}/{timeframe}/{pair}_{timeframe}.parquet" #
    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S") #
    output_gcs_path = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}/{pair}_{timeframe}_recent.parquet" #
    logger.info(f"Initializing prepare_opt_data.py for {pair} {timeframe} (Input: {input_gcs_path}, Output: {output_gcs_path})") #
    command = ["python", "prepare_opt_data.py", "--input_path", input_gcs_path, "--output_path", output_gcs_path] #
    process = subprocess.run(command, capture_output=True, text=True, check=False) #

    if process.returncode != 0:
        logger.error(f"Error in prepare_opt_data.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}") #
        raise RuntimeError(f"prepare_opt_data.py failed for {pair} {timeframe}") #
    else:
        logger.info(f"prepare_opt_data.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}") #
    return output_gcs_path #


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, #
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow", 
        "optuna", "scikit-learn", "joblib", 
        "kfp-pipeline-spec>=0.1.16" 
    ] #
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
) -> str: #
    import subprocess
    from datetime import datetime
    import json 
    import logging
    from pathlib import Path 

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") #
    logger = logging.getLogger(__name__)

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S") #
    params_output_dir = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}" #
    final_params_gcs_path = f"{params_output_dir}/best_params.json" #

    logger.info(f"Initializing optimize_lstm.py for {pair} {timeframe} (Features: {features_path}, Params Output Dir: {params_output_dir}, Final Params File: {final_params_gcs_path}, Trials: {n_trials})") #

    command = [
        "python", "optimize_lstm.py",
        "--features", features_path,
        "--pair", pair,
        "--timeframe", timeframe,
        "--output", final_params_gcs_path, 
        "--n-trials", str(n_trials),
    ] #
    logger.info(f"Executing command: {' '.join(command)}") #
    process = subprocess.run(command, capture_output=True, text=True, check=False) #

    if process.returncode != 0:
        error_msg = f"Error in optimize_lstm.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}" #
        logger.error(error_msg) #
        raise RuntimeError(error_msg) #
    else:
        logger.info(f"optimize_lstm.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}") #
        try:
            best_score_line = None #
            for line in process.stdout.splitlines():
                if "Best is trial" in line and "with value:" in line: #
                    best_score_line = line #
                    break 
            if best_score_line:
                parts = best_score_line.split("with value:") #
                if len(parts) > 1:
                    score_str = parts[1].split(" ")[1].replace(".", "") 
                    score_value_str = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', score_str.split()[0])) #
                    score_value = float(score_value_str) #
                    optimization_metrics.log_metric("optuna_best_trial_score", score_value) #
                    logger.info(f"Optuna best trial score metric logged: {score_value}") #
            else:
                logger.warning("Could not find Optuna best trial score in stdout for metrics logging.") #
        except Exception as e:
            logger.warning(f"Error parsing Optuna best trial score: {e}. Metrics may not be fully logged. Full stdout was:\n{process.stdout}") #

    return final_params_gcs_path #


@dsl.component(
    base_image="python:3.9-slim", 
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"] #
)
def train_lstm_vertex_ai_op(
    project_id: str,
    region: str,
    gcs_bucket_name: str, 
    params_path: str, 
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, 
    vertex_training_image_uri: str, 
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
) -> str: 
    import logging
    import time
    from datetime import datetime, timezone
    from google.cloud import aiplatform as gcp_aiplatform
    from google.cloud import storage as gcp_storage

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") #
    logger = logging.getLogger(__name__)

    staging_gcs_path = f"gs://{gcs_bucket_name}/staging_for_custom_jobs" #
    logger.info(f"Configurando staging bucket para Vertex AI Custom Job: {staging_gcs_path}") #

    gcp_aiplatform.init(project=project_id, location=region, staging_bucket=staging_gcs_path) #
    logger.info(f"Vertex AI SDK inicializado. Project: {project_id}, Region: {region}, Staging Bucket: {staging_gcs_path}") #

    timestamp_for_job = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") #
    job_display_name = f"lstm-training-{pair.lower()}-{timeframe.lower()}-{timestamp_for_job}" #

    training_script_args = [
        "--params", params_path, 
        "--output-gcs-base-dir", output_gcs_prefix, 
        "--pair", pair,
        "--timeframe", timeframe,
        "--project-id", project_id,
        "--gcs-bucket-name", gcs_bucket_name,
    ] #

    current_accelerator_type = vertex_accelerator_type #
    current_accelerator_count = vertex_accelerator_count #

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": vertex_machine_type,
        },
        "replica_count": 1, #
        "container_spec": {
            "image_uri": vertex_training_image_uri, #
            "args": training_script_args, #
            "command": ["python", "train_lstm.py"], 
        },
    }] #

    if current_accelerator_count > 0 and current_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED": #
        worker_pool_specs[0]["machine_spec"]["accelerator_type"] = current_accelerator_type #
        worker_pool_specs[0]["machine_spec"]["accelerator_count"] = current_accelerator_count #
        logger.info(f"Configurando Custom Job con GPU: {current_accelerator_count} x {current_accelerator_type}") #
    else:
        logger.info(f"Configurando Custom Job solo con CPU.") #

    vertex_job_base_output_dir = f"gs://{gcs_bucket_name}/vertex_ai_job_outputs/{job_display_name}" #
    logger.info(f"Submitting Vertex AI Custom Job: {job_display_name} with args: {training_script_args}") #
    logger.info(f"Vertex AI Custom Job base output directory: {vertex_job_base_output_dir}") #

    custom_job = gcp_aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=vertex_job_base_output_dir, 
        project=project_id,
        location=region,
    ) #
    custom_job_launch_time_utc = datetime.utcnow().replace(tzinfo=timezone.utc) #
    try:
        custom_job.run(service_account=vertex_service_account, sync=True, timeout=10800) #
        logger.info(f"Vertex AI Custom Job {custom_job.display_name} (ID: {custom_job.resource_name}) completed.") #
    except Exception as e:
        logger.error(f"Vertex AI Custom Job {job_display_name} failed or timed out: {e}") #
        if custom_job and custom_job.resource_name:
             logger.error(f"Job details: {custom_job.resource_name}, state: {custom_job.state}") #
        raise RuntimeError(f"Vertex AI Custom Job for LSTM training failed: {e}") #

    if not output_gcs_prefix.startswith("gs://"): #
        raise ValueError("output_gcs_prefix debe ser una ruta GCS (gs://...)") #

    prefix_parts = output_gcs_prefix.replace("gs://", "").split("/") #
    expected_bucket_name = prefix_parts[0] #
    gcs_listing_prefix = f"{'/'.join(prefix_parts[1:])}/{pair}/{timeframe}/" #
    if not gcs_listing_prefix.endswith('/'): gcs_listing_prefix += '/' #

    logger.info(f"Polling GCS bucket '{expected_bucket_name}' under prefix '{gcs_listing_prefix}' for model artifacts created by the training script (after {custom_job_launch_time_utc.isoformat()}).") #

    storage_client = gcp_storage.Client(project=project_id) #
    bucket = storage_client.bucket(expected_bucket_name) #
    max_poll_wait_sec, poll_interval_sec = 15 * 60, 30 
    start_poll_time = time.time() #
    found_model_dir_gcs_path = None #

    while time.time() - start_poll_time < max_poll_wait_sec: #
        logger.info(f"GCS poll elapsed: {int(time.time() - start_poll_time)}s / {max_poll_wait_sec}s. Checking prefix: gs://{expected_bucket_name}/{gcs_listing_prefix}") #
        blobs = bucket.list_blobs(prefix=gcs_listing_prefix, delimiter="/") #
        candidate_dirs = [] #
        for page in blobs.pages: 
            if hasattr(page, 'prefixes') and page.prefixes: #
                for dir_prefix in page.prefixes: 
                    model_blob_path = f"{dir_prefix.rstrip('/')}/model.h5" #
                    if bucket.blob(model_blob_path).exists(): #
                        try:
                            dir_name_part = dir_prefix.strip('/').split('/')[-1] #
                            datetime.strptime(dir_name_part, "%Y%m%d%H%M%S") 
                            candidate_dirs.append(f"gs://{expected_bucket_name}/{dir_prefix.rstrip('/')}") #
                        except ValueError:
                            logger.warning(f"Directory {dir_prefix} does not seem to be a timestamped model directory. Skipping.") #
                            continue 
        
        if candidate_dirs:
            found_model_dir_gcs_path = sorted(candidate_dirs, reverse=True)[0] #
            logger.info(f"Found LSTM model directory in GCS: {found_model_dir_gcs_path}") #
            return found_model_dir_gcs_path.rstrip('/') 
        
        time.sleep(poll_interval_sec) #

    err_msg = f"LSTM model directory (containing model.h5) not found in gs://{expected_bucket_name}/{gcs_listing_prefix} after {max_poll_wait_sec / 60:.1f} min. Vertex Job: {custom_job.display_name}" #
    logger.error(err_msg) #
    raise TimeoutError(err_msg) #


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, 
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "joblib", 
    ] #
)
def prepare_rl_data_op(
    gcs_bucket_name: str, 
    lstm_model_dir: str, 
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, 
    project_id: str, 
) -> str: 
    import subprocess
    from datetime import datetime
    import logging
    from pathlib import Path 

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") #
    logger = logging.getLogger(__name__)

    if not lstm_model_dir or not lstm_model_dir.startswith("gs://"): #
        raise ValueError(f"prepare_rl_data_op requiere una ruta GCS válida para lstm_model_dir, se obtuvo: {lstm_model_dir}") #

    lstm_model_path = f"{lstm_model_dir.rstrip('/')}/model.h5" #
    lstm_scaler_path = f"{lstm_model_dir.rstrip('/')}/scaler.pkl" #
    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json" #

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S") #
    rl_data_output_path = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}/ppo_input_data.npz" #

    logger.info(f"Initializing prepare_rl_data.py for {pair} {timeframe}. LSTM model dir: {lstm_model_dir}. Output NPZ: {rl_data_output_path}") #
    command = [
        "python", "prepare_rl_data.py",
        "--model", lstm_model_path,
        "--scaler", lstm_scaler_path,
        "--params", lstm_params_path, 
        "--output", rl_data_output_path,
        "--pair", pair,
        "--timeframe", timeframe,
        "--gcs-bucket-name", gcs_bucket_name, 
    ] #
    logger.info(f"Executing command: {' '.join(command)}") #
    process = subprocess.run(command, capture_output=True, text=True, check=False) #

    if process.returncode != 0:
        error_msg = f"Error in prepare_rl_data.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}" #
        logger.error(error_msg) #
        raise RuntimeError(error_msg) #
    else:
        logger.info(f"prepare_rl_data.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}") #

    return rl_data_output_path #


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, 
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "stable-baselines3[extra]", "gymnasium", "pandas_ta", "scikit-learn",
        "joblib", "optuna", 
    ] #
)
def train_rl_op(
    gcs_bucket_name: str, 
    lstm_model_dir: str, 
    rl_data_path: str, 
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, 
    tensorboard_logs_gcs_prefix: str, 
    project_id: str, 
) -> str: 
    import subprocess
    from datetime import datetime
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") #
    logger = logging.getLogger(__name__)

    if not lstm_model_dir or not lstm_model_dir.startswith("gs://"): #
        raise ValueError(f"train_rl_op requiere una ruta GCS válida para lstm_model_dir, se obtuvo: {lstm_model_dir}") #

    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json" #
    tensorboard_log_gcs_path = f"{tensorboard_logs_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}" #

    logger.info(f"Initializing train_rl.py for {pair} {timeframe}. LSTM params: {lstm_params_path}, RL Data: {rl_data_path}, Output Base: {output_gcs_prefix}, TensorBoard: {tensorboard_log_gcs_path}") #
    command = [
        "python", "train_rl.py",
        "--params", lstm_params_path, 
        "--rl-data", rl_data_path, 
        "--output-bucket", gcs_bucket_name, 
        "--pair", pair,
        "--timeframe", timeframe,
        "--output-model-base-gcs-path", output_gcs_prefix, 
        "--tensorboard-log-dir", tensorboard_log_gcs_path,
    ] #
    logger.info(f"Executing command: {' '.join(command)}") #
    process = subprocess.run(command, capture_output=True, text=True, check=False) #

    if process.returncode != 0:
        error_msg = f"Error in train_rl.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}" #
        logger.error(error_msg) #
        raise RuntimeError(error_msg) #
    else:
        logger.info(f"train_rl.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}") #
        rl_model_upload_line = next((l for l in process.stdout.splitlines() if "RL model saved to gs://" in l or "Subido ppo_filter_model.zip a" in l), None) #
        if rl_model_upload_line:
            if "RL model saved to " in rl_model_upload_line: #
                 uploaded_rl_path = rl_model_upload_line.split("RL model saved to ")[1].strip() #
            elif "Subido ppo_filter_model.zip a " in rl_model_upload_line: #
                 uploaded_rl_path = rl_model_upload_line.split("Subido ppo_filter_model.zip a ")[1].strip() #
            else: 
                logger.warning("Could not precisely determine RL model path from stdout using known patterns. Looking for any 'gs://' path.") #
                gs_paths_in_stdout = [word for word in process.stdout.split() if word.startswith("gs://") and word.endswith(".zip")] #
                if gs_paths_in_stdout:
                    uploaded_rl_path = gs_paths_in_stdout[-1] #
                    logger.info(f"Guessed RL model path from stdout: {uploaded_rl_path}") #
                else:
                    raise RuntimeError("Failed to determine RL model output path from train_rl.py stdout. No 'gs://...zip' path found.") #

            logger.info(f"RL model uploaded to: {uploaded_rl_path}") #
            return uploaded_rl_path.rstrip('/') #
        else:
            raise RuntimeError("Failed to determine RL model output path from train_rl.py stdout. Ensure the script prints the GCS path.") #


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, 
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "stable-baselines3", "gymnasium", "pandas_ta", "scikit-learn",
        "joblib", "optuna", 
        "kfp-pipeline-spec>=0.1.16" 
    ] #
)
def backtest_op(
    lstm_model_dir: str, 
    rl_model_path: str, 
    features_path: str, 
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, 
    project_id: str,
    backtest_metrics: Output[Metrics] 
) -> str: 
    import subprocess
    from datetime import datetime
    import logging
    import json as py_json 
    from google.cloud import storage as gcp_storage
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") #
    logger = logging.getLogger(__name__)

    if not lstm_model_dir.startswith("gs://") or not rl_model_path.startswith("gs://"): #
        raise ValueError("backtest_op requiere rutas GCS válidas para los modelos.") #

    lstm_model_file_path = f"{lstm_model_dir.rstrip('/')}/model.h5" #
    lstm_scaler_path = f"{lstm_model_dir.rstrip('/')}/scaler.pkl" #
    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json" #

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S") #
    backtest_output_gcs_dir = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}/" #
    expected_metrics_file_gcs_path = f"{backtest_output_gcs_dir.rstrip('/')}/metrics.json" #

    logger.info(f"Initializing backtest.py for {pair} {timeframe}. Output dir: {backtest_output_gcs_dir}") #
    command = [
        "python", "backtest.py",
        "--pair", pair,
        "--timeframe", timeframe,
        "--lstm-model-path", lstm_model_file_path,
        "--lstm-scaler-path", lstm_scaler_path,
        "--lstm-params-path", lstm_params_path,
        "--rl-model-path", rl_model_path,
        "--features-path", features_path, 
        "--output-dir", backtest_output_gcs_dir, 
    ] #
    logger.info(f"Executing command: {' '.join(command)}") #
    process = subprocess.run(command, capture_output=True, text=True, check=False) #

    if process.returncode != 0:
        error_msg = f"Error in backtest.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}" #
        logger.error(error_msg) #
        raise RuntimeError(error_msg) #
    else:
        logger.info(f"backtest.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}") #
        try:
            storage_client = gcp_storage.Client(project=project_id) #
            path_parts = expected_metrics_file_gcs_path.replace("gs://", "").split("/", 1) #
            bucket_name_str = path_parts[0] #
            blob_path_str = path_parts[1] #

            bucket = storage_client.bucket(bucket_name_str) #
            blob = bucket.blob(blob_path_str) #

            if blob.exists(): #
                logger.info(f"Attempting to download metrics from: {expected_metrics_file_gcs_path}") #
                metrics_content = blob.download_as_string() #
                metrics_data = py_json.loads(metrics_content) #
                logger.info(f"Metrics data loaded: {metrics_data}") #
                for key, value in metrics_data.items():
                    if isinstance(value, (int, float)): #
                        backtest_metrics.log_metric(key, value) #
                    else:
                        try: 
                            float_value = float(value) #
                            backtest_metrics.log_metric(key, float_value) #
                        except (ValueError, TypeError):
                            logger.warning(f"Cannot log metric '{key}' with value '{value}' (type: {type(value)}) as float. Skipping or consider logging as metadata.") #
                logger.info(f"Logged metrics from {expected_metrics_file_gcs_path} to KFP Metrics.") #
            else:
                logger.warning(f"Metrics file {expected_metrics_file_gcs_path} not found for KFP logging. STDOUT from backtest.py might contain clues:\n{process.stdout}") #
        except Exception as e_metrics:
            logger.error(f"Failed to log metrics from {expected_metrics_file_gcs_path} to KFP: {e_metrics}", exc_info=True) #

    return backtest_output_gcs_dir.rstrip('/') #


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, 
    packages_to_install=["google-cloud-storage", "gcsfs", "pandas", "numpy"] 
) #
def decide_promotion_op(
    new_backtest_metrics_dir: str, 
    new_lstm_artifacts_dir: str, 
    new_rl_model_path: str, 
    gcs_bucket_name: str, 
    pair: str,
    timeframe: str,
    project_id: str, 
    current_production_metrics_path: str, 
    production_base_dir: str, 
) -> bool: 
    import subprocess
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") #
    logger = logging.getLogger(__name__)

    new_metrics_file_path = f"{new_backtest_metrics_dir.rstrip('/')}/metrics.json" #

    if not new_lstm_artifacts_dir.startswith("gs://") or \
       not new_rl_model_path.startswith("gs://") or \
       not new_metrics_file_path.startswith("gs://"): #
        raise ValueError("decide_promotion_op requiere rutas GCS válidas para artefactos y métricas.") #

    production_pair_timeframe_dir = f"{production_base_dir.rstrip('/')}/{pair}/{timeframe}" #

    logger.info(f"Initializing model promotion decision for {pair} {timeframe}.") #
    logger.info(f"  New metrics file: {new_metrics_file_path}") #
    logger.info(f"  Current prod metrics: {current_production_metrics_path}") #
    logger.info(f"  New LSTM dir: {new_lstm_artifacts_dir}") #
    logger.info(f"  New RL model: {new_rl_model_path}") #
    logger.info(f"  Production target dir: {production_pair_timeframe_dir}") #

    command = [
        "python", "model_promotion_decision.py",
        "--new-metrics-path", new_metrics_file_path,
        "--current-production-metrics-path", current_production_metrics_path,
        "--new-lstm-artifacts-dir", new_lstm_artifacts_dir, 
        "--new-rl-model-path", new_rl_model_path, 
        "--production-pair-timeframe-dir", production_pair_timeframe_dir, 
        "--pair", pair, 
        "--timeframe", timeframe, 
    ] #
    logger.info(f"Executing command: {' '.join(command)}") #
    process = subprocess.run(command, capture_output=True, text=True, check=False) #

    model_promoted_flag = False #
    if process.returncode == 0:
        logger.info(f"model_promotion_decision.py completed.\nSTDOUT: {process.stdout}") #
        if "PROMOVIDO a producción." in process.stdout or "Model promoted to production" in process.stdout.lower(): #
            model_promoted_flag = True #
            logger.info("Model promotion confirmed based on script output.") #
        else:
            logger.info("Model not promoted based on script output.") #
    else:
        logger.error(f"Error in model_promotion_decision.py. Model will not be promoted.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}") #

    logger.info(f"Model promoted: {model_promoted_flag}") #
    return model_promoted_flag #


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, 
    packages_to_install=["google-cloud-pubsub"], #
)
def notify_pipeline_status_op(
    project_id: str,
    pair: str,
    timeframe: str,
    pipeline_run_status: str, 
    pipeline_job_id: str, 
    pipeline_job_link: str, 
    model_promoted: bool = False, 
) -> bool: 
    from google.cloud import pubsub_v1
    import logging
    import json as py_json
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") #
    logger = logging.getLogger(__name__)

    SUCCESS_TOPIC_ID = "data-ingestion-success" 
    FAILURE_TOPIC_ID = "data-ingestion-failures" 

    try:
        publisher = pubsub_v1.PublisherClient() #
        ts_utc_iso = datetime.now(timezone.utc).isoformat() #

        status_summary = pipeline_run_status.upper() #
        if status_summary not in ["SUCCESS", "FAILURE", "SUCCEEDED", "FAILED", "CANCELLED"]: 
            logger.warning(f"Pipeline status '{pipeline_run_status}' no es canónico. Se normalizará para el mensaje.") #
            if "fail" in pipeline_run_status.lower() or "error" in pipeline_run_status.lower(): #
                status_summary = "FAILURE" #
            elif "success" in pipeline_run_status.lower() or "succeeded" in pipeline_run_status.lower(): #
                status_summary = "SUCCESS" #
            else: 
                status_summary = "UNKNOWN_STATUS" 

        details = {
            "pipeline_name": "algo-trading-mlops-gcp-pipeline-v2", 
            "pipeline_job_id": pipeline_job_id, #
            "pipeline_job_link": pipeline_job_link, #
            "pair": pair, #
            "timeframe": timeframe, #
            "run_status": status_summary, 
            "model_promoted_status": model_promoted if status_summary == "SUCCESS" else False, 
            "timestamp_utc": ts_utc_iso, #
        } #

        target_topic_id = SUCCESS_TOPIC_ID if status_summary == "SUCCESS" else FAILURE_TOPIC_ID #
        if status_summary == "UNKNOWN_STATUS": 
             target_topic_id = FAILURE_TOPIC_ID 

        summary_message = f"MLOps Pipeline Notification: {details['pipeline_name']} for {pair}/{timeframe} finished with status: {status_summary}." #
        if status_summary == "SUCCESS": #
            summary_message += f" Model Promoted: {model_promoted}." #

        msg_data = {"summary": summary_message, "details": details} #
        msg_bytes = py_json.dumps(msg_data, indent=2).encode("utf-8") #
        topic_path = publisher.topic_path(project_id, target_topic_id) #

        future = publisher.publish(topic_path, msg_bytes) #
        message_id = future.result(timeout=60) #
        logger.info(f"Notification sent to Pub/Sub topic '{topic_path}' with message ID: {message_id}.") #
        return True #
    except Exception as e:
        logger.error(f"Error publishing notification to Pub/Sub: {e}", exc_info=True) #
        return False #


# --- Pipeline Definition ---
@dsl.pipeline(
    name="algo-trading-mlops-gcp-pipeline-v2", #
    description="KFP v2 Pipeline for training and deploying algorithmic trading models.", #
    pipeline_root=PIPELINE_ROOT, #
)
def training_pipeline(
    pair: str = DEFAULT_PAIR, #
    timeframe: str = DEFAULT_TIMEFRAME, #
    n_trials: int = DEFAULT_N_TRIALS, #
    backtest_features_path: str = f"gs://{GCS_BUCKET_NAME}/backtest_data/{DEFAULT_PAIR}_{DEFAULT_TIMEFRAME}_unseen.parquet", #
    vertex_lstm_training_image: str = VERTEX_LSTM_TRAINER_IMAGE_URI, #
    vertex_lstm_machine_type: str = DEFAULT_VERTEX_LSTM_MACHINE_TYPE, #
    vertex_lstm_accelerator_type: str = DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE, #
    vertex_lstm_accelerator_count: int = DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT, #
    vertex_lstm_service_account: str = DEFAULT_VERTEX_LSTM_SERVICE_ACCOUNT, #
    kfp_opt_accelerator_type: str = KFP_COMPONENT_ACCELERATOR_TYPE, #
    kfp_opt_accelerator_count: int = KFP_COMPONENT_ACCELERATOR_COUNT, #
    polygon_api_key_secret_name_param: str = "polygon-api-key", #
    polygon_api_key_secret_version_param: str = "latest", #
):
    # CORRECCIÓN: Usar nombres de variables Python consistentes y placeholders KFP correctos
    pipeline_name_val = dsl.PIPELINE_JOB_NAME_PLACEHOLDER
    pipeline_job_id_val = dsl.PIPELINE_JOB_ID_PLACEHOLDER
    pipeline_ui_link_val = f"https://console.cloud.google.com/vertex-ai/pipelines/runs/{pipeline_job_id_val}?project={PROJECT_ID}" #

    with dsl.ExitHandler(
        notify_pipeline_status_op(
            project_id=PROJECT_ID,        # ID de tu proyecto
            pair=pair,                    # Par de divisas
            timeframe=timeframe,          # Timeframe
            pipeline_run_status="{{$.pipeline_task_status}}",  # Estado del pipeline
            pipeline_job_id=pipeline_job_id_val,               # ID del job
            pipeline_job_link=pipeline_ui_link_val,            # Enlace en la UI
            model_promoted=False          # Valor por defecto; puede fallar el pipeline
        )

    ):
        update_data_task = update_data_op(
            pair=pair,
            timeframe=timeframe,
            gcs_bucket_name=GCS_BUCKET_NAME,
            project_id=PROJECT_ID,
            polygon_api_key_secret_name=polygon_api_key_secret_name_param,
            polygon_api_key_secret_version=polygon_api_key_secret_version_param
        ) #
        update_data_task.set_display_name("Update_Market_Data") #
        update_data_task.set_cpu_limit("2").set_memory_limit("4G") #

        prepare_opt_data_task = prepare_opt_data_op(
            gcs_bucket_name=GCS_BUCKET_NAME, #
            pair=pair, #
            timeframe=timeframe, #
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/data_filtered_for_opt_v2", 
            project_id=PROJECT_ID #
        ).after(update_data_task) #
        prepare_opt_data_task.set_display_name("Prepare_Optimization_Data") #
        prepare_opt_data_task.set_cpu_limit("4").set_memory_limit("15G") #

        optimize_lstm_task = optimize_lstm_op(
            gcs_bucket_name=GCS_BUCKET_NAME, 
            features_path=prepare_opt_data_task.output, 
            pair=pair, #
            timeframe=timeframe, #
            n_trials=n_trials, #
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/params/LSTM_v2", 
            project_id=PROJECT_ID #
        ) #
        optimize_lstm_task.set_display_name("Optimize_LSTM_Hyperparameters") #
        optimize_lstm_task.set_cpu_limit("8").set_memory_limit("30G") #

        train_lstm_task = train_lstm_vertex_ai_op(
            project_id=PROJECT_ID, #
            region=REGION, #
            gcs_bucket_name=GCS_BUCKET_NAME, #
            params_path=optimize_lstm_task.outputs['Output'], 
            pair=pair, #
            timeframe=timeframe, #
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/models/LSTM_v2", 
            vertex_training_image_uri=vertex_lstm_training_image, #
            vertex_machine_type=vertex_lstm_machine_type, #
            vertex_accelerator_type=vertex_lstm_accelerator_type, #
            vertex_accelerator_count=vertex_lstm_accelerator_count, #
            vertex_service_account=vertex_lstm_service_account #
        ) #
        train_lstm_task.set_display_name("Train_LSTM_Model_Vertex_AI") #
        train_lstm_task.set_cpu_limit("1").set_memory_limit("2G") #

        prepare_rl_data_task = prepare_rl_data_op(
            gcs_bucket_name=GCS_BUCKET_NAME, #
            lstm_model_dir=train_lstm_task.output, #
            pair=pair, #
            timeframe=timeframe, #
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/data_rl_inputs_v2", 
            project_id=PROJECT_ID #
        ) #
        prepare_rl_data_task.set_display_name("Prepare_RL_Data") #
        prepare_rl_data_task.set_cpu_limit("8").set_memory_limit("30G") #

        train_rl_task = train_rl_op(
            gcs_bucket_name=GCS_BUCKET_NAME, #
            lstm_model_dir=train_lstm_task.output, #
            rl_data_path=prepare_rl_data_task.output, #
            pair=pair, #
            timeframe=timeframe, #
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/models/RL_v2", 
            tensorboard_logs_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/tensorboard_logs_v2", 
            project_id=PROJECT_ID #
        ) #
        train_rl_task.set_display_name("Train_PPO_Agent") #
        train_rl_task.set_cpu_limit("16").set_memory_limit("60G") #

        backtest_task = backtest_op(
            lstm_model_dir=train_lstm_task.output, #
            rl_model_path=train_rl_task.output, #
            features_path=backtest_features_path, #
            pair=pair, #
            timeframe=timeframe, #
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/backtest_results_v2", 
            project_id=PROJECT_ID #
        ) #
        backtest_task.set_display_name("Execute_Full_Backtesting") #
        backtest_task.set_cpu_limit("8").set_memory_limit("30G") #

        decide_task = decide_promotion_op(
            new_backtest_metrics_dir=backtest_task.outputs['Output'], 
            new_lstm_artifacts_dir=train_lstm_task.output, #
            new_rl_model_path=train_rl_task.output, #
            gcs_bucket_name=GCS_BUCKET_NAME, #
            pair=pair, #
            timeframe=timeframe, #
            project_id=PROJECT_ID, #
            current_production_metrics_path=f"gs://{GCS_BUCKET_NAME}/models/production_v2/{pair}/{timeframe}/metrics_production.json", 
            production_base_dir=f"gs://{GCS_BUCKET_NAME}/models/production_v2" 
        ) #
        decide_task.set_display_name("Decide_Model_Promotion") #
        decide_task.set_cpu_limit("2").set_memory_limit("4G") #

        # CORRECCIÓN: Usar dsl.If y pasar variables correctas para ID y Link
        with dsl.If(decide_task.output == True, name="If_Model_Promoted"): #
            notify_promoted_task = notify_pipeline_status_op(
                project_id=PROJECT_ID, #
                pair=pair, #
                timeframe=timeframe, #
                pipeline_run_status="SUCCESS_PROMOTED", 
                model_promoted=True, #
                pipeline_job_id=pipeline_job_id_val,
                pipeline_job_link=pipeline_ui_link_val,
            ) #
            notify_promoted_task.set_display_name("Notify_Pipeline_Success_Promoted") #

        with dsl.If(decide_task.output == False, name="If_Model_Not_Promoted"): #
            notify_not_promoted_task = notify_pipeline_status_op(
                project_id=PROJECT_ID, #
                pair=pair, #
                timeframe=timeframe, #
                pipeline_run_status="SUCCESS_NOT_PROMOTED", 
                model_promoted=False, #
                pipeline_job_id=pipeline_job_id_val,
                pipeline_job_link=pipeline_ui_link_val,
            ) #
            notify_not_promoted_task.set_display_name("Notify_Pipeline_Success_Not_Promoted") #


# --- Pipeline Compilation and Optional Direct Execution ---
if __name__ == "__main__":
    pipeline_filename = "algo_trading_mlops_gcp_pipeline_v2.json" #
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=pipeline_filename
    ) #
    print(f"✅ KFP Pipeline (compatible con v2) compilado a {pipeline_filename}") #

    SUBMIT_TO_VERTEX_AI = os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() != "false"

    if SUBMIT_TO_VERTEX_AI: #
        print(f"\n🚀 Iniciando la sumisión y ejecución directa de la pipeline '{pipeline_filename}' en Vertex AI...") #

        aiplatform_sdk.init(project=PROJECT_ID, location=REGION) #
        print(f" Vertex AI SDK inicializado para Project: {PROJECT_ID}, Region: {REGION}") #

        run_pair = os.getenv("PIPELINE_PAIR", DEFAULT_PAIR) #
        run_timeframe = os.getenv("PIPELINE_TIMEFRAME", DEFAULT_TIMEFRAME) #
        run_n_trials = int(os.getenv("PIPELINE_N_TRIALS", str(DEFAULT_N_TRIALS))) #
        run_backtest_features_path = os.getenv(
            "PIPELINE_BACKTEST_FEATURES_PATH",
            f"gs://{GCS_BUCKET_NAME}/backtest_data/{run_pair}_{run_timeframe}_unseen.parquet"
        ) #
        run_polygon_secret_name = os.getenv("PIPELINE_POLYGON_SECRET_NAME", "polygon-api-key") #

        pipeline_parameter_values = {
            "pair": run_pair, #
            "timeframe": run_timeframe, #
            "n_trials": run_n_trials, #
            "backtest_features_path": run_backtest_features_path, #
            "vertex_lstm_training_image": VERTEX_LSTM_TRAINER_IMAGE_URI, #
            "vertex_lstm_machine_type": DEFAULT_VERTEX_LSTM_MACHINE_TYPE, #
            "vertex_lstm_accelerator_type": DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE, #
            "vertex_lstm_accelerator_count": DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT, #
            "vertex_lstm_service_account": DEFAULT_VERTEX_LSTM_SERVICE_ACCOUNT, #
            "kfp_opt_accelerator_type": KFP_COMPONENT_ACCELERATOR_TYPE, #
            "kfp_opt_accelerator_count": KFP_COMPONENT_ACCELERATOR_COUNT, #
            "polygon_api_key_secret_name_param": run_polygon_secret_name, #
            "polygon_api_key_secret_version_param": "latest", 
        } #
        # Corregir el typo en el nombre del parámetro si existe
        if "kfp_opt_ accelerator_type" in pipeline_parameter_values: #
            pipeline_parameter_values["kfp_opt_accelerator_type"] = pipeline_parameter_values.pop("kfp_opt_ accelerator_type") #

        job_display_name = f"algo-trading-v2-{run_pair}-{run_timeframe}-{datetime.now().strftime('%Y%m%d%H%M%S')}" #

        print(f" Display Name para PipelineJob: {job_display_name}") #
        print(f" Template Path (archivo JSON compilado): {pipeline_filename}") #
        print(f" Pipeline Root: {PIPELINE_ROOT}") #
        print(" Pipeline Parameter Values:") #
        print(json.dumps(pipeline_parameter_values, indent=2)) #

        job = aiplatform_sdk.PipelineJob(
            display_name=job_display_name, #
            template_path=pipeline_filename, #
            pipeline_root=PIPELINE_ROOT, #
            location=REGION, #
            enable_caching=False, #
            parameter_values=pipeline_parameter_values, #
        ) #

        job_resource_name = None #

        try:
            print(f"\nEnviando PipelineJob '{job_display_name}' y esperando su finalización (ejecución síncrona)...") #
            job.run() #
            job_resource_name = job.resource_name #

            print(f"\n✅ PipelineJob completado.") #
            print(f" Job Resource Name: {job.resource_name}") #
            if hasattr(job, 'state') and job.state: #
                 print(f" Estado final del Job: {job.state}") #
            if job.error: #
                print(f" Job Error (si existe): {job.error}") #

        except Exception as e:
            print(f"\n❌ Error durante la sumisión o ejecución del PipelineJob:") #
            print(f"  Tipo de Excepción: {type(e).__name__}") #
            print(f"  Mensaje: {str(e)}") #
            print("  Traceback:") #
            traceback.print_exc() #

            if job_resource_name: #
                 print(f"  Job Resource Name (si se obtuvo antes del error): {job_resource_name}") #
                 try:
                     job.refresh() #
                     current_job_state = job.state if hasattr(job, 'state') else "No disponible" #
                     current_job_error = job.error if hasattr(job, 'error') else "No disponible" #
                     print(f"  Estado actual del Job (refrescado, si disponible): {current_job_state}") #
                     print(f"  Error del Job (refrescado, si disponible): {current_job_error}") #
                 except Exception as refresh_err:
                     print(f"  No se pudo refrescar el estado del job: {refresh_err}") #
            else:
                print("  No se pudo obtener el Job Resource Name (falló la sumisión inicial o la creación del objeto Job).") #
        finally:
            if job_resource_name: #
                job_id_for_link = job_resource_name.split('/')[-1] #
                pipeline_console_link = f"https://console.cloud.google.com/vertex-ai/pipelines/runs/{job_id_for_link}?project={PROJECT_ID}" #
                print(f"\n🔗 Visualizar en Vertex AI Pipelines: {pipeline_console_link}") #
            else:
                print("\nℹ️ No se generó un enlace de ejecución ya que el PipelineJob no se pudo someter o falló muy temprano.") #
    else:
        print(f"Pipeline compilado a {pipeline_filename}. Para ejecutar, sube este archivo a Vertex AI Pipelines o establece SUBMIT_PIPELINE_TO_VERTEX=true.") #