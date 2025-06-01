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

# KFP imports (Corregidas para eliminar DeprecationWarning)
from kfp import compiler, dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact

# --- Global Project and Bucket Configuration ---
PROJECT_ID = "trading-ai-460823"
REGION = "us-central1"

GCS_BUCKET_NAME = "trading-ai-models-460823"
PIPELINE_ROOT = f"gs://{GCS_BUCKET_NAME}/pipeline_root_v2"

# --- Docker Image URIs ---
KFP_COMPONENTS_IMAGE_URI = (
    f"europe-west1-docker.pkg.dev/{PROJECT_ID}/data-ingestion-repo/data-ingestion-agent:latest"
)
VERTEX_LSTM_TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/trading-images/trainer-cpu:latest"
)

# --- Machine Types and GPUs for Vertex AI (Configurado para CPU por ahora) ---
DEFAULT_VERTEX_LSTM_MACHINE_TYPE = "n1-standard-4"
DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT = 0

KFP_COMPONENT_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
KFP_COMPONENT_ACCELERATOR_COUNT = 0

# --- Service Account for Vertex AI Custom Jobs ---
DEFAULT_VERTEX_LSTM_SERVICE_ACCOUNT = (
    f"vertex-custom-job-sa@{PROJECT_ID}.iam.gserviceaccount.com"
)

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
        "requests", # <--- AÑADIDO para obtener info de la SA
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
    import requests # <--- AÑADIDO
    from google.cloud import secretmanager as sm_client_lib

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s [%(funcName)s]: %(message)s")
    logger = logging.getLogger(__name__)

    # --- INICIO: Loguear identidad de la cuenta de servicio ---
    try:
        metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"
        headers = {"Metadata-Flavor": "Google"}
        sa_email_response = requests.get(metadata_url, headers=headers, timeout=5) # Timeout corto
        sa_email_response.raise_for_status() # Lanza excepción si hay error HTTP
        current_sa = sa_email_response.text
        logger.info(f"El componente update-data-op se está ejecutando con la cuenta de servicio (desde metadata): {current_sa}")
    except Exception as sa_err:
        logger.warning(f"No se pudo obtener la cuenta de servicio actual del metadata server: {sa_err}")
        logger.warning("Esto es normal si se ejecuta localmente fuera de GCP. En GCP, podría indicar un problema de red/configuración del metadata server.")
    # --- FIN: Loguear identidad de la cuenta de servicio ---

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
        logger.error(f"Error al obtener la API Key de Polygon de Secret Manager: {e}", exc_info=True) # exc_info=True para traceback completo
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
        if stdout_output:
            error_message += f"\nSTDOUT:\n{stdout_output}"
        if stderr_output:
            error_message += f"\nSTDERR:\n{stderr_output}"

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
    import subprocess
    from datetime import datetime
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    input_gcs_path = f"gs://{gcs_bucket_name}/data/{pair}/{timeframe}/{pair}_{timeframe}.parquet"
    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # Corregir la construcción de output_gcs_path para que sea un archivo, no un prefijo
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
    base_image=KFP_COMPONENTS_IMAGE_URI, # Debería ser VERTEX_LSTM_TRAINER_IMAGE_URI si optimize_lstm.py usa TF y otras libs pesadas
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow", # Asegúrate que TF esté aquí si no está en la base_image
        "optuna", "scikit-learn", "joblib", # Otras dependencias de optimize_lstm.py
        "kfp-pipeline-spec>=0.1.16" # Necesario para Output[Metrics]
    ]
)
def optimize_lstm_op(
    gcs_bucket_name: str, # No se usa directamente en el comando, pero puede ser útil para logs o artefactos intermedios
    features_path: str, # Este es el Input[Dataset].uri que se pasa a --features
    pair: str,
    timeframe: str,
    n_trials: int,
    output_gcs_prefix: str, # Directorio base donde se guardará best_params.json
    project_id: str, # No se usa directamente en el comando, pero puede ser útil
    optimization_metrics: Output[Metrics] # Para loggear métricas de Optuna
) -> str: # Retorna la ruta GCS al archivo best_params.json
    import subprocess
    from datetime import datetime
    import json # Para parsear stdout si es necesario
    import logging
    from pathlib import Path # Para construir rutas

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # El archivo de parámetros se guardará en un subdirectorio con timestamp dentro del prefijo de salida
    params_output_dir = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}"
    # El nombre del archivo de salida de optimize_lstm.py es fijo (best_params.json)
    # y el script lo guardará dentro del directorio que le pasemos a --output
    # Entonces, --output debe ser params_output_dir
    # La función retornará la ruta completa al archivo best_params.json
    final_params_gcs_path = f"{params_output_dir}/best_params.json"

    logger.info(f"Initializing optimize_lstm.py for {pair} {timeframe} (Features: {features_path}, Params Output Dir: {params_output_dir}, Final Params File: {final_params_gcs_path}, Trials: {n_trials})")

    # El argumento --output para optimize_lstm.py espera la RUTA COMPLETA al archivo .json
    # Corregido: optimize_lstm.py ahora espera la RUTA COMPLETA al archivo json, no un directorio.
    command = [
        "python", "optimize_lstm.py",
        "--features", features_path,
        "--pair", pair,
        "--timeframe", timeframe,
        "--output", final_params_gcs_path, # Pasar la ruta completa del archivo
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
        # Intentar extraer la métrica del estudio de Optuna si se imprime en stdout
        # Ejemplo: "Best trial score: 123.45"
        try:
            # Buscar una línea que contenga "Best is trial" o similar, que Optuna imprime
            # Ejemplo de log de Optuna: "[I 2023-10-27 10:00:00,000] Trial 2 finished with value: 123.45 and parameters: {'lr': 0.001}. Best is trial 2 with value: 123.45."
            best_score_line = None
            for line in process.stdout.splitlines():
                if "Best is trial" in line and "with value:" in line:
                    best_score_line = line
                    break # Tomar la última (o la primera que encuentre si el formato es consistente)

            if best_score_line:
                # Extraer el valor numérico
                parts = best_score_line.split("with value:")
                if len(parts) > 1:
                    score_str = parts[1].split(" ")[1].replace(".", "") # "123.45." -> "12345" (si tiene punto al final)
                    # Asegurar que solo tomamos el número
                    score_value_str = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', score_str.split()[0]))
                    score_value = float(score_value_str)
                    optimization_metrics.log_metric("optuna_best_trial_score", score_value)
                    logger.info(f"Optuna best trial score metric logged: {score_value}")
            else:
                logger.warning("Could not find Optuna best trial score in stdout for metrics logging.")
        except Exception as e:
            logger.warning(f"Error parsing Optuna best trial score: {e}. Metrics may not be fully logged. Full stdout was:\n{process.stdout}")

    # Retornar la ruta GCS al archivo de parámetros generado
    return final_params_gcs_path


@dsl.component(
    base_image="python:3.9-slim", # Imagen base ligera para el lanzador del job
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"]
)
def train_lstm_vertex_ai_op(
    project_id: str,
    region: str,
    gcs_bucket_name: str, # Bucket principal para artefactos
    params_path: str, # Ruta GCS al best_params.json de Optuna
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, # Prefijo GCS para guardar el modelo entrenado (directorio)
    vertex_training_image_uri: str, # Imagen Docker para el CustomJob de entrenamiento
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
) -> str: # Retorna la ruta GCS al directorio del modelo guardado
    import logging
    import time
    from datetime import datetime, timezone
    from google.cloud import aiplatform as gcp_aiplatform
    from google.cloud import storage as gcp_storage

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # --- MODIFICACIÓN AQUÍ (Opción 1) ---
    # Define la ruta GCS para el staging bucket.
    # Puedes usar una subcarpeta dentro del bucket principal que ya estás utilizando.
    staging_gcs_path = f"gs://{gcs_bucket_name}/staging_for_custom_jobs"
    logger.info(f"Configurando staging bucket para Vertex AI Custom Job: {staging_gcs_path}")

    # Inicializa el SDK de Vertex AI incluyendo el staging_bucket
    gcp_aiplatform.init(project=project_id, location=region, staging_bucket=staging_gcs_path)
    logger.info(f"Vertex AI SDK inicializado. Project: {project_id}, Region: {region}, Staging Bucket: {staging_gcs_path}")
    # --- FIN DE MODIFICACIÓN ---

    timestamp_for_job = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_display_name = f"lstm-training-{pair.lower()}-{timeframe.lower()}-{timestamp_for_job}"

    # El script train_lstm.py esperará un --output-gcs-base-dir donde creará
    # un subdirectorio con timestamp para guardar los artefactos del modelo.
    # La ruta base que se pasa aquí será donde el script de entrenamiento cree su propia subcarpeta.
    training_script_args = [
        "--params", params_path, # Ruta al best_params.json
        "--output-gcs-base-dir", output_gcs_prefix, # Directorio base para la salida del modelo
        "--pair", pair,
        "--timeframe", timeframe,
        # project_id y gcs_bucket_name pueden ser necesarios para que el script de entrenamiento
        # descargue datos adicionales o guarde artefactos intermedios si es necesario.
        "--project-id", project_id,
        "--gcs-bucket-name", gcs_bucket_name,
    ]

    current_accelerator_type = vertex_accelerator_type
    current_accelerator_count = vertex_accelerator_count

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": vertex_machine_type,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": vertex_training_image_uri,
            "args": training_script_args,
            "command": ["python", "train_lstm.py"], # Asumiendo que el script se llama train_lstm.py
        },
    }]

    if current_accelerator_count > 0 and current_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        worker_pool_specs[0]["machine_spec"]["accelerator_type"] = current_accelerator_type
        worker_pool_specs[0]["machine_spec"]["accelerator_count"] = current_accelerator_count
        logger.info(f"Configurando Custom Job con GPU: {current_accelerator_count} x {current_accelerator_type}")
    else:
        logger.info(f"Configurando Custom Job solo con CPU.")

    # El base_output_dir para el CustomJob es donde Vertex AI guarda logs y otros artefactos del job.
    # No es necesariamente donde el script de entrenamiento guarda el modelo.
    vertex_job_base_output_dir = f"gs://{gcs_bucket_name}/vertex_ai_job_outputs/{job_display_name}"
    logger.info(f"Submitting Vertex AI Custom Job: {job_display_name} with args: {training_script_args}")
    logger.info(f"Vertex AI Custom Job base output directory: {vertex_job_base_output_dir}")

    custom_job = gcp_aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=vertex_job_base_output_dir, # Directorio para logs y artefactos del job de Vertex
        project=project_id,
        location=region,
    )
    custom_job_launch_time_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    try:
        # El timeout es para el job de Vertex AI, no para el polling de GCS.
        custom_job.run(service_account=vertex_service_account, sync=True, timeout=10800) # Sincrónico, espera 3 horas
        logger.info(f"Vertex AI Custom Job {custom_job.display_name} (ID: {custom_job.resource_name}) completed.")
    except Exception as e:
        logger.error(f"Vertex AI Custom Job {job_display_name} failed or timed out: {e}")
        if custom_job and custom_job.resource_name:
             logger.error(f"Job details: {custom_job.resource_name}, state: {custom_job.state}")
        # Re-lanzar la excepción para que el pipeline de KFP marque el paso como fallido
        raise RuntimeError(f"Vertex AI Custom Job for LSTM training failed: {e}")

    # --- Polling para el directorio del modelo ---
    # El script train_lstm.py debe crear un subdirectorio con timestamp dentro de output_gcs_prefix.
    # Ejemplo: gs://<bucket>/models/LSTM_v2/<pair>/<timeframe>/<timestamp_del_script>/
    if not output_gcs_prefix.startswith("gs://"):
        raise ValueError("output_gcs_prefix debe ser una ruta GCS (gs://...)")

    prefix_parts = output_gcs_prefix.replace("gs://", "").split("/")
    expected_bucket_name = prefix_parts[0]
    # La base para listar es el output_gcs_prefix, donde train_lstm.py crea subdirectorios.
    # Asumiendo que train_lstm.py crea <output_gcs_prefix>/<pair>/<timeframe>/<timestamp_from_script>/model.h5
    gcs_listing_prefix = f"{'/'.join(prefix_parts[1:])}/{pair}/{timeframe}/"
    if not gcs_listing_prefix.endswith('/'): gcs_listing_prefix += '/'

    logger.info(f"Polling GCS bucket '{expected_bucket_name}' under prefix '{gcs_listing_prefix}' for model artifacts created by the training script (after {custom_job_launch_time_utc.isoformat()}).")

    storage_client = gcp_storage.Client(project=project_id)
    bucket = storage_client.bucket(expected_bucket_name)
    max_poll_wait_sec, poll_interval_sec = 15 * 60, 30 # 15 minutos de espera máxima
    start_poll_time = time.time()
    found_model_dir_gcs_path = None

    while time.time() - start_poll_time < max_poll_wait_sec:
        logger.info(f"GCS poll elapsed: {int(time.time() - start_poll_time)}s / {max_poll_wait_sec}s. Checking prefix: gs://{expected_bucket_name}/{gcs_listing_prefix}")
        # Listar "directorios" (prefijos comunes) directamente bajo gcs_listing_prefix
        blobs = bucket.list_blobs(prefix=gcs_listing_prefix, delimiter="/")
        candidate_dirs = []
        # blobs.prefixes contendrá los nombres de los subdirectorios con timestamp creados por el script
        for page in blobs.pages: # Necesario para iterar sobre todos los resultados si hay muchos
            if hasattr(page, 'prefixes') and page.prefixes:
                for dir_prefix in page.prefixes: # dir_prefix es algo como 'models/LSTM_v2/EURUSD/15minute/20230101120000/'
                    # Verificar si model.h5 existe en este directorio candidato
                    # dir_prefix ya incluye el gcs_listing_prefix implícitamente por cómo se listó
                    # El blob_name para verificar debe ser relativo al bucket
                    model_blob_path = f"{dir_prefix.rstrip('/')}/model.h5"
                    if bucket.blob(model_blob_path).exists():
                        # Validar que el nombre del directorio sea un timestamp (opcional pero bueno)
                        try:
                            dir_name_part = dir_prefix.strip('/').split('/')[-1]
                            datetime.strptime(dir_name_part, "%Y%m%d%H%M%S") # Asume este formato del script de training
                            # Construir la ruta GCS completa al directorio del modelo
                            candidate_dirs.append(f"gs://{expected_bucket_name}/{dir_prefix.rstrip('/')}")
                        except ValueError:
                            logger.warning(f"Directory {dir_prefix} does not seem to be a timestamped model directory. Skipping.")
                            continue # No es un directorio de modelo esperado
        
        if candidate_dirs:
            # Ordenar por el nombre del directorio (timestamp) en orden descendente para obtener el más reciente
            # Esto asume que los nombres de directorio son timestamps que se pueden ordenar alfabéticamente
            found_model_dir_gcs_path = sorted(candidate_dirs, reverse=True)[0]
            logger.info(f"Found LSTM model directory in GCS: {found_model_dir_gcs_path}")
            return found_model_dir_gcs_path.rstrip('/') # Asegurar que no haya / al final
        
        time.sleep(poll_interval_sec)

    err_msg = f"LSTM model directory (containing model.h5) not found in gs://{expected_bucket_name}/{gcs_listing_prefix} after {max_poll_wait_sec / 60:.1f} min. Vertex Job: {custom_job.display_name}"
    logger.error(err_msg)
    raise TimeoutError(err_msg)


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, # O la imagen que tenga las dependencias de prepare_rl_data.py
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "joblib", # Para scaler.pkl
        # Añadir otras dependencias específicas de prepare_rl_data.py si no están en la base_image
    ]
)
def prepare_rl_data_op(
    gcs_bucket_name: str, # No se usa directamente en el comando, pero puede ser útil
    lstm_model_dir: str, # Ruta GCS al directorio del modelo LSTM (salida de train_lstm_vertex_ai_op)
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, # Prefijo GCS para guardar los datos de entrada de RL
    project_id: str, # No se usa directamente, pero puede ser útil
) -> str: # Retorna la ruta GCS al archivo .npz generado
    import subprocess
    from datetime import datetime
    import logging
    from pathlib import Path # Para construir rutas

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if not lstm_model_dir or not lstm_model_dir.startswith("gs://"):
        raise ValueError(f"prepare_rl_data_op requiere una ruta GCS válida para lstm_model_dir, se obtuvo: {lstm_model_dir}")

    # Construir rutas a los artefactos del modelo LSTM dentro del directorio proporcionado
    lstm_model_path = f"{lstm_model_dir.rstrip('/')}/model.h5"
    lstm_scaler_path = f"{lstm_model_dir.rstrip('/')}/scaler.pkl"
    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json" # Asumiendo que params.json también se guarda aquí

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # El archivo de salida .npz se guardará en un subdirectorio con timestamp
    rl_data_output_path = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}/ppo_input_data.npz"

    logger.info(f"Initializing prepare_rl_data.py for {pair} {timeframe}. LSTM model dir: {lstm_model_dir}. Output NPZ: {rl_data_output_path}")
    command = [
        "python", "prepare_rl_data.py",
        "--model", lstm_model_path,
        "--scaler", lstm_scaler_path,
        "--params", lstm_params_path, # Necesita el params.json que contiene la configuración original de Optuna, incluyendo 'win'
        "--output", rl_data_output_path,
        # Pasar pair y timeframe si el script los necesita para encontrar datos de mercado originales
        "--pair", pair,
        "--timeframe", timeframe,
        "--gcs-bucket-name", gcs_bucket_name, # Para que el script pueda acceder a los datos de mercado originales
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
    base_image=KFP_COMPONENTS_IMAGE_URI, # O la imagen que tenga las dependencias de train_rl.py (SB3, Gym, etc.)
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "stable-baselines3[extra]", "gymnasium", "pandas_ta", "scikit-learn",
        "joblib", "optuna", # Optuna podría no ser necesaria aquí si los params ya están resueltos
    ]
)
def train_rl_op(
    gcs_bucket_name: str, # Bucket principal
    lstm_model_dir: str, # Ruta GCS al directorio del modelo LSTM (para params.json, etc.)
    rl_data_path: str, # Ruta GCS al archivo .npz de entrada (salida de prepare_rl_data_op)
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, # Prefijo GCS para guardar el modelo RL entrenado (directorio)
    tensorboard_logs_gcs_prefix: str, # Prefijo GCS para logs de TensorBoard
    project_id: str, # No se usa directamente, pero puede ser útil
) -> str: # Retorna la ruta GCS al archivo del modelo RL (.zip)
    import subprocess
    from datetime import datetime
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if not lstm_model_dir or not lstm_model_dir.startswith("gs://"):
        raise ValueError(f"train_rl_op requiere una ruta GCS válida para lstm_model_dir, se obtuvo: {lstm_model_dir}")

    # El script train_rl.py podría necesitar el params.json del LSTM para algunas configuraciones.
    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json"

    # El script train_rl.py debe manejar la creación de su propio subdirectorio con timestamp
    # para guardar el modelo RL, similar a como lo hace train_lstm.py.
    # El output_gcs_prefix es el directorio base donde el script guardará su salida.
    tensorboard_log_gcs_path = f"{tensorboard_logs_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    logger.info(f"Initializing train_rl.py for {pair} {timeframe}. LSTM params: {lstm_params_path}, RL Data: {rl_data_path}, Output Base: {output_gcs_prefix}, TensorBoard: {tensorboard_log_gcs_path}")
    command = [
        "python", "train_rl.py",
        "--params", lstm_params_path, # Pasar el params.json del LSTM
        "--rl-data", rl_data_path, # El archivo .npz
        "--output-bucket", gcs_bucket_name, # Bucket para guardar el modelo RL
        "--pair", pair,
        "--timeframe", timeframe,
        "--output-model-base-gcs-path", output_gcs_prefix, # Directorio base para el modelo RL
        "--tensorboard-log-dir", tensorboard_log_gcs_path,
        # Potencialmente, el scaler del LSTM podría ser útil si el entorno RL necesita alguna transformación similar
        # "--lstm-scaler-path", f"{lstm_model_dir.rstrip('/')}/scaler.pkl",
    ]
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        error_msg = f"Error in train_rl.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        logger.info(f"train_rl.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}")
        # Extraer la ruta del modelo RL guardado desde stdout
        # Asumir que el script train_rl.py imprime una línea como:
        # "RL model saved to gs://<bucket>/<output_gcs_prefix>/<pair>/<timeframe>/<timestamp>/ppo_model.zip"
        rl_model_upload_line = next((l for l in process.stdout.splitlines() if "RL model saved to gs://" in l or "Subido ppo_filter_model.zip a" in l), None)
        if rl_model_upload_line:
            # Extraer la ruta GCS
            if "RL model saved to " in rl_model_upload_line:
                 uploaded_rl_path = rl_model_upload_line.split("RL model saved to ")[1].strip()
            elif "Subido ppo_filter_model.zip a " in rl_model_upload_line:
                 uploaded_rl_path = rl_model_upload_line.split("Subido ppo_filter_model.zip a ")[1].strip()
            else: # Fallback si el formato del log cambia
                logger.warning("Could not precisely determine RL model path from stdout using known patterns. Looking for any 'gs://' path.")
                gs_paths_in_stdout = [word for word in process.stdout.split() if word.startswith("gs://") and word.endswith(".zip")]
                if gs_paths_in_stdout:
                    uploaded_rl_path = gs_paths_in_stdout[-1] # Tomar el último como el más probable
                    logger.info(f"Guessed RL model path from stdout: {uploaded_rl_path}")
                else:
                    raise RuntimeError("Failed to determine RL model output path from train_rl.py stdout. No 'gs://...zip' path found.")

            logger.info(f"RL model uploaded to: {uploaded_rl_path}")
            return uploaded_rl_path.rstrip('/')
        else:
            raise RuntimeError("Failed to determine RL model output path from train_rl.py stdout. Ensure the script prints the GCS path.")


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, # O la imagen que tenga las dependencias de backtest.py
    packages_to_install=[
        "google-cloud-storage", "gcsfs", "pandas", "numpy", "tensorflow",
        "stable-baselines3", "gymnasium", "pandas_ta", "scikit-learn",
        "joblib", "optuna", # Optuna puede no ser necesaria, pero otras sí
        "kfp-pipeline-spec>=0.1.16" # Para Output[Metrics]
    ]
)
def backtest_op(
    lstm_model_dir: str, # Ruta GCS al directorio del modelo LSTM
    rl_model_path: str, # Ruta GCS al archivo .zip del modelo RL
    features_path: str, # Ruta GCS al parquet con datos de backtesting (unseen)
    pair: str,
    timeframe: str,
    output_gcs_prefix: str, # Prefijo GCS para guardar los resultados del backtest (directorio)
    project_id: str,
    backtest_metrics: Output[Metrics] # Para loggear métricas del backtest
) -> str: # Retorna la ruta GCS al directorio de resultados del backtest
    import subprocess
    from datetime import datetime
    import logging
    import json as py_json # Para leer el archivo metrics.json
    from google.cloud import storage as gcp_storage
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if not lstm_model_dir.startswith("gs://") or not rl_model_path.startswith("gs://"):
        raise ValueError("backtest_op requiere rutas GCS válidas para los modelos.")

    # Construir rutas a los artefactos del modelo LSTM
    lstm_model_file_path = f"{lstm_model_dir.rstrip('/')}/model.h5"
    lstm_scaler_path = f"{lstm_model_dir.rstrip('/')}/scaler.pkl"
    lstm_params_path = f"{lstm_model_dir.rstrip('/')}/params.json"

    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # El script de backtesting guardará sus artefactos (incluyendo metrics.json) en este directorio
    backtest_output_gcs_dir = f"{output_gcs_prefix.rstrip('/')}/{pair}/{timeframe}/{timestamp_str}/"
    # La ruta esperada al archivo de métricas que generará el script
    expected_metrics_file_gcs_path = f"{backtest_output_gcs_dir.rstrip('/')}/metrics.json"

    logger.info(f"Initializing backtest.py for {pair} {timeframe}. Output dir: {backtest_output_gcs_dir}")
    command = [
        "python", "backtest.py",
        "--pair", pair,
        "--timeframe", timeframe,
        "--lstm-model-path", lstm_model_file_path,
        "--lstm-scaler-path", lstm_scaler_path,
        "--lstm-params-path", lstm_params_path,
        "--rl-model-path", rl_model_path,
        "--features-path", features_path, # Datos de backtesting
        "--output-dir", backtest_output_gcs_dir, # Directorio donde guardar los resultados
    ]
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        error_msg = f"Error in backtest.py for {pair} {timeframe}.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        logger.info(f"backtest.py completed for {pair} {timeframe}.\nSTDOUT: {process.stdout}")
        # Leer el archivo metrics.json de GCS y loggear las métricas a KFP
        try:
            storage_client = gcp_storage.Client(project=project_id)
            # Quitar 'gs://' y luego dividir bucket y blob path
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
                for key, value in metrics_data.items():
                    if isinstance(value, (int, float)):
                        backtest_metrics.log_metric(key, value)
                    else:
                        try: # Intentar convertir a float si es posible (e.g., string numérico)
                            float_value = float(value)
                            backtest_metrics.log_metric(key, float_value)
                        except (ValueError, TypeError):
                            logger.warning(f"Cannot log metric '{key}' with value '{value}' (type: {type(value)}) as float. Skipping or consider logging as metadata.")
                logger.info(f"Logged metrics from {expected_metrics_file_gcs_path} to KFP Metrics.")
            else:
                logger.warning(f"Metrics file {expected_metrics_file_gcs_path} not found for KFP logging. STDOUT from backtest.py might contain clues:\n{process.stdout}")
        except Exception as e_metrics:
            logger.error(f"Failed to log metrics from {expected_metrics_file_gcs_path} to KFP: {e_metrics}", exc_info=True)

    return backtest_output_gcs_dir.rstrip('/')


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, # O la imagen que tenga las dependencias de model_promotion_decision.py
    packages_to_install=["google-cloud-storage", "gcsfs", "pandas", "numpy"] # Asegurar pandas si se leen métricas como CSV/JSON
)
def decide_promotion_op(
    new_backtest_metrics_dir: str, # Directorio de salida de backtest_op
    new_lstm_artifacts_dir: str, # Directorio de salida de train_lstm_vertex_ai_op
    new_rl_model_path: str, # Archivo .zip de salida de train_rl_op
    gcs_bucket_name: str, # No se usa directamente, pero podría ser útil
    pair: str,
    timeframe: str,
    project_id: str, # No se usa directamente, pero podría ser útil
    current_production_metrics_path: str, # Ruta GCS al metrics.json del modelo en producción
    production_base_dir: str, # Directorio base en GCS para los modelos de producción
) -> bool: # Retorna True si se promueve, False en caso contrario
    import subprocess
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # La ruta al archivo de métricas del nuevo modelo
    new_metrics_file_path = f"{new_backtest_metrics_dir.rstrip('/')}/metrics.json"

    if not new_lstm_artifacts_dir.startswith("gs://") or \
       not new_rl_model_path.startswith("gs://") or \
       not new_metrics_file_path.startswith("gs://"):
        raise ValueError("decide_promotion_op requiere rutas GCS válidas para artefactos y métricas.")

    # Directorio de producción específico para este par y timeframe
    production_pair_timeframe_dir = f"{production_base_dir.rstrip('/')}/{pair}/{timeframe}"

    logger.info(f"Initializing model promotion decision for {pair} {timeframe}.")
    logger.info(f"  New metrics file: {new_metrics_file_path}")
    logger.info(f"  Current prod metrics: {current_production_metrics_path}")
    logger.info(f"  New LSTM dir: {new_lstm_artifacts_dir}")
    logger.info(f"  New RL model: {new_rl_model_path}")
    logger.info(f"  Production target dir: {production_pair_timeframe_dir}")

    command = [
        "python", "model_promotion_decision.py",
        "--new-metrics-path", new_metrics_file_path,
        "--current-production-metrics-path", current_production_metrics_path,
        "--new-lstm-artifacts-dir", new_lstm_artifacts_dir, # Directorio del modelo LSTM candidato
        "--new-rl-model-path", new_rl_model_path, # Archivo del modelo RL candidato
        "--production-pair-timeframe-dir", production_pair_timeframe_dir, # Donde copiar si se promueve
        "--pair", pair, # Para logging o lógica interna del script
        "--timeframe", timeframe, # Para logging o lógica interna del script
    ]
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    model_promoted_flag = False
    if process.returncode == 0:
        logger.info(f"model_promotion_decision.py completed.\nSTDOUT: {process.stdout}")
        # El script debe indicar claramente si promovió o no en su stdout
        if "PROMOVIDO a producción." in process.stdout or "Model promoted to production" in process.stdout.lower():
            model_promoted_flag = True
            logger.info("Model promotion confirmed based on script output.")
        else:
            logger.info("Model not promoted based on script output.")
    else:
        # Si el script falla, no se promueve. El error ya se loguea.
        logger.error(f"Error in model_promotion_decision.py. Model will not be promoted.\nSTDOUT: {process.stdout}\nSTDERR: {process.stderr}")

    logger.info(f"Model promoted: {model_promoted_flag}")
    return model_promoted_flag


@dsl.component(
    base_image=KFP_COMPONENTS_IMAGE_URI, # Imagen base con gcloud SDK o pubsub client
    packages_to_install=["google-cloud-pubsub"],
)
def notify_pipeline_status_op(
    project_id: str,
    pair: str,
    timeframe: str,
    pipeline_run_status: str, # "SUCCESS" o "FAILURE" (o estado del pipeline job)
    pipeline_job_id: str, # ID del PipelineJob de Vertex AI
    pipeline_job_link: str, # Enlace a la consola de Vertex AI Pipelines
    model_promoted: bool = False, # Salida de decide_promotion_op
    # Podrías añadir más detalles como las rutas de los artefactos si es necesario
    # new_lstm_model_dir: str = "",
    # new_rl_model_path: str = "",
    # backtest_results_dir: str = "",
) -> bool: # Retorna True si la notificación fue exitosa
    from google.cloud import pubsub_v1
    import logging
    import json as py_json
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Definir IDs de tópicos de Pub/Sub (deben existir en tu proyecto GCP)
    SUCCESS_TOPIC_ID = "data-ingestion-success" # Renombrar a algo como "pipeline-notifications-success"
    FAILURE_TOPIC_ID = "data-ingestion-failures" # Renombrar a algo como "pipeline-notifications-failure"

    try:
        publisher = pubsub_v1.PublisherClient()
        ts_utc_iso = datetime.now(timezone.utc).isoformat()

        status_summary = pipeline_run_status.upper()
        if status_summary not in ["SUCCESS", "FAILURE", "SUCCEEDED", "FAILED", "CANCELLED"]: # Otros estados de Vertex AI
            logger.warning(f"Pipeline status '{pipeline_run_status}' no es canónico. Se normalizará para el mensaje.")
            if "fail" in pipeline_run_status.lower() or "error" in pipeline_run_status.lower():
                status_summary = "FAILURE"
            elif "success" in pipeline_run_status.lower() or "succeeded" in pipeline_run_status.lower():
                status_summary = "SUCCESS"
            else: # Otros estados como "CANCELLED", "PENDING", "RUNNING"
                status_summary = "UNKNOWN_STATUS" # O manejar según necesidad

        details = {
            "pipeline_name": "algo-trading-mlops-gcp-pipeline-v2", # Podrías hacerlo un parámetro
            "pipeline_job_id": pipeline_job_id,
            "pipeline_job_link": pipeline_job_link,
            "pair": pair,
            "timeframe": timeframe,
            "run_status": status_summary, # Usar el estado normalizado
            "model_promoted_status": model_promoted if status_summary == "SUCCESS" else False, # Solo relevante si éxito
            "timestamp_utc": ts_utc_iso,
            # "lstm_model_location": new_lstm_model_dir if status_summary == "SUCCESS" else "N/A",
            # "rl_model_location": new_rl_model_path if status_summary == "SUCCESS" else "N/A",
            # "backtest_results_location": backtest_results_dir if status_summary == "SUCCESS" else "N/A",
        }

        target_topic_id = SUCCESS_TOPIC_ID if status_summary == "SUCCESS" else FAILURE_TOPIC_ID
        if status_summary == "UNKNOWN_STATUS": # Decidir a dónde enviar si el estado no es final
             target_topic_id = FAILURE_TOPIC_ID # O un tópico de "warnings/info"

        summary_message = f"MLOps Pipeline Notification: {details['pipeline_name']} for {pair}/{timeframe} finished with status: {status_summary}."
        if status_summary == "SUCCESS":
            summary_message += f" Model Promoted: {model_promoted}."

        msg_data = {"summary": summary_message, "details": details}
        msg_bytes = py_json.dumps(msg_data, indent=2).encode("utf-8") # indent para legibilidad en logs de PubSub
        topic_path = publisher.topic_path(project_id, target_topic_id)

        future = publisher.publish(topic_path, msg_bytes)
        # Esperar a que se publique el mensaje (con timeout)
        message_id = future.result(timeout=60)
        logger.info(f"Notification sent to Pub/Sub topic '{topic_path}' with message ID: {message_id}.")
        return True
    except Exception as e:
        logger.error(f"Error publishing notification to Pub/Sub: {e}", exc_info=True)
        return False


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
    # Ruta a los datos de backtesting (unseen data)
    backtest_features_path: str = f"gs://{GCS_BUCKET_NAME}/backtest_data/{DEFAULT_PAIR}_{DEFAULT_TIMEFRAME}_unseen.parquet",
    vertex_lstm_training_image: str = VERTEX_LSTM_TRAINER_IMAGE_URI,
    vertex_lstm_machine_type: str = DEFAULT_VERTEX_LSTM_MACHINE_TYPE,
    vertex_lstm_accelerator_type: str = DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE,
    vertex_lstm_accelerator_count: int = DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT,
    vertex_lstm_service_account: str = DEFAULT_VERTEX_LSTM_SERVICE_ACCOUNT,
    # Parámetros para GPU en componentes KFP (actualmente no usado para Optuna en el código)
    kfp_opt_accelerator_type: str = KFP_COMPONENT_ACCELERATOR_TYPE,
    kfp_opt_accelerator_count: int = KFP_COMPONENT_ACCELERATOR_COUNT,
    # Nombres de secretos en Secret Manager
    polygon_api_key_secret_name_param: str = "polygon-api-key",
    polygon_api_key_secret_version_param: str = "latest",
):
    # Obtener el ID del Pipeline Job y el enlace para notificaciones
    # Estas son variables de contexto de KFP que se resuelven en tiempo de ejecución
    kfp_pipeline_name = dsl.PIPELINE_JOB_NAME_PLACEHOLDER
    kfp_run_id = dsl.PIPELINE_RUN_ID_PLACEHOLDER # Este es el ID corto del run
    # El job_id completo (resource name) puede ser más difícil de obtener directamente aquí para el link
    # Una aproximación para el link (puede necesitar ajuste según la URL de tu consola GCP)
    kfp_ui_link = f"https://console.cloud.google.com/vertex-ai/pipelines/runs/{kfp_run_id}?project={PROJECT_ID}"


    # --- Tareas del Pipeline ---
    with dsl.ExitHandler(
        exit_op=notify_pipeline_status_op( # Componente a ejecutar al salir
            project_id=PROJECT_ID,
            pair=pair,
            timeframe=timeframe,
            # {{$.pipeline_task_status}} se resuelve al estado final del pipeline ('Succeeded', 'Failed', 'Cancelled', 'Skipped')
            pipeline_run_status=dsl.PIPELINE_TASK_STATUS_PLACEHOLDER,
            pipeline_job_id=kfp_run_id, # Usar el run ID
            pipeline_job_link=kfp_ui_link,
            # model_promoted se pasará desde el flujo principal si tiene éxito
        )
    ):
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
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/data_filtered_for_opt_v2", # Prefijo para el archivo parquet
            project_id=PROJECT_ID
        ).after(update_data_task)
        prepare_opt_data_task.set_display_name("Prepare_Optimization_Data")
        prepare_opt_data_task.set_cpu_limit("4").set_memory_limit("15G")

        optimize_lstm_task = optimize_lstm_op(
            gcs_bucket_name=GCS_BUCKET_NAME, # Pasado para consistencia, aunque no usado directamente por el script
            features_path=prepare_opt_data_task.output, # Salida es la ruta al archivo parquet
            pair=pair,
            timeframe=timeframe,
            n_trials=n_trials,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/params/LSTM_v2", # Prefijo para el best_params.json
            project_id=PROJECT_ID
        )
        optimize_lstm_task.set_display_name("Optimize_LSTM_Hyperparameters")
        # Ajustar recursos según necesidad; Optuna puede ser intensivo
        optimize_lstm_task.set_cpu_limit("8").set_memory_limit("30G") # Ejemplo, podría necesitar más o menos
        # Configuración de GPU comentada, ya que no se usa activamente en el script de optimización
        # if kfp_opt_accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED" and kfp_opt_accelerator_count > 0:
        #     optimize_lstm_task.set_gpu_limit(str(kfp_opt_accelerator_count)) # kfp v1 style
        #     optimize_lstm_task.add_node_selector_constraint(
        #         'cloud.google.com/gke-accelerator', kfp_opt_accelerator_type # kfp v1 style
        #     )

        train_lstm_task = train_lstm_vertex_ai_op(
            project_id=PROJECT_ID,
            region=REGION,
            gcs_bucket_name=GCS_BUCKET_NAME,
            params_path=optimize_lstm_task.outputs['Output'], # Salida es la ruta al best_params.json
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/models/LSTM_v2", # Prefijo para el directorio del modelo
            vertex_training_image_uri=vertex_lstm_training_image,
            vertex_machine_type=vertex_lstm_machine_type,
            vertex_accelerator_type=vertex_lstm_accelerator_type,
            vertex_accelerator_count=vertex_lstm_accelerator_count,
            vertex_service_account=vertex_lstm_service_account
        )
        train_lstm_task.set_display_name("Train_LSTM_Model_Vertex_AI")
        # Este componente es un lanzador, no necesita muchos recursos
        train_lstm_task.set_cpu_limit("1").set_memory_limit("2G")


        prepare_rl_data_task = prepare_rl_data_op(
            gcs_bucket_name=GCS_BUCKET_NAME,
            lstm_model_dir=train_lstm_task.output, # Salida es la ruta al directorio del modelo LSTM
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/data_rl_inputs_v2", # Prefijo para el archivo .npz
            project_id=PROJECT_ID
        )
        prepare_rl_data_task.set_display_name("Prepare_RL_Data")
        prepare_rl_data_task.set_cpu_limit("8").set_memory_limit("30G")


        train_rl_task = train_rl_op(
            gcs_bucket_name=GCS_BUCKET_NAME,
            lstm_model_dir=train_lstm_task.output, # Directorio del modelo LSTM (para params.json)
            rl_data_path=prepare_rl_data_task.output, # Archivo .npz
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/models/RL_v2", # Prefijo para el modelo RL .zip
            tensorboard_logs_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/tensorboard_logs_v2", # Prefijo para logs de TB
            project_id=PROJECT_ID
        )
        train_rl_task.set_display_name("Train_PPO_Agent")
        # El entrenamiento de RL puede ser muy intensivo
        train_rl_task.set_cpu_limit("16").set_memory_limit("60G") # Ajustar según sea necesario


        backtest_task = backtest_op(
            lstm_model_dir=train_lstm_task.output, # Directorio del modelo LSTM
            rl_model_path=train_rl_task.output, # Archivo .zip del modelo RL
            features_path=backtest_features_path, # Datos de backtesting (unseen)
            pair=pair,
            timeframe=timeframe,
            output_gcs_prefix=f"gs://{GCS_BUCKET_NAME}/backtest_results_v2", # Prefijo para resultados de backtest
            project_id=PROJECT_ID
        )
        backtest_task.set_display_name("Execute_Full_Backtesting")
        backtest_task.set_cpu_limit("8").set_memory_limit("30G")


        decide_task = decide_promotion_op(
            new_backtest_metrics_dir=backtest_task.outputs['Output'], # Directorio de resultados del backtest
            new_lstm_artifacts_dir=train_lstm_task.output, # Directorio del modelo LSTM
            new_rl_model_path=train_rl_task.output, # Archivo .zip del modelo RL
            gcs_bucket_name=GCS_BUCKET_NAME,
            pair=pair,
            timeframe=timeframe,
            project_id=PROJECT_ID,
            current_production_metrics_path=f"gs://{GCS_BUCKET_NAME}/models/production_v2/{pair}/{timeframe}/metrics_production.json", # Ajustar si es necesario
            production_base_dir=f"gs://{GCS_BUCKET_NAME}/models/production_v2" # Directorio base para producción
        )
        decide_task.set_display_name("Decide_Model_Promotion")
        decide_task.set_cpu_limit("2").set_memory_limit("4G")

        # Notificación de éxito (se ejecuta solo si el pipeline llega hasta aquí sin fallar antes del ExitHandler)
        # Esta notificación se enviará ANTES de que el ExitHandler se ejecute si el pipeline tiene éxito.
        # El ExitHandler se ejecutará de todas formas al final.
        # Para evitar doble notificación de éxito, se puede condicionar esta o la del ExitHandler.
        # Por simplicidad, el ExitHandler manejará todos los estados finales.
        # Si se quiere una notificación específica de promoción *dentro* del flujo normal:
        with dsl.Condition(decide_task.output == True, name="If_Model_Promoted_Notify_Success"):
            notify_promoted_task = notify_pipeline_status_op(
                project_id=PROJECT_ID, pair=pair, timeframe=timeframe,
                pipeline_run_status="SUCCESS_PROMOTED", # Estado personalizado para distinguir
                model_promoted=True,
                pipeline_job_id=kfp_run_id,
                pipeline_job_link=kfp_ui_link,
            )
            notify_promoted_task.set_display_name("Notify_Pipeline_Success_Promoted")

        with dsl.Condition(decide_task.output == False, name="If_Model_Not_Promoted_Notify_Success"):
            notify_not_promoted_task = notify_pipeline_status_op(
                project_id=PROJECT_ID, pair=pair, timeframe=timeframe,
                pipeline_run_status="SUCCESS_NOT_PROMOTED", # Estado personalizado
                model_promoted=False,
                pipeline_job_id=kfp_run_id,
                pipeline_job_link=kfp_ui_link,
            )
            notify_not_promoted_task.set_display_name("Notify_Pipeline_Success_Not_Promoted")


# --- Pipeline Compilation and Optional Direct Execution ---
if __name__ == "__main__":
    pipeline_filename = "algo_trading_mlops_gcp_pipeline_v2.json"
    # Usar kfp.compiler.Compiler() para KFP v2
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=pipeline_filename
    )
    print(f"✅ KFP Pipeline (compatible con v2) compilado a {pipeline_filename}")

    # --- Configuración para la sumisión directa a Vertex AI ---
    SUBMIT_TO_VERTEX_AI = os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "false").lower() == "true"

    if SUBMIT_TO_VERTEX_AI:
        print(f"\n🚀 Iniciando la sumisión y ejecución directa de la pipeline '{pipeline_filename}' en Vertex AI...")

        aiplatform_sdk.init(project=PROJECT_ID, location=REGION) # Staging bucket se define en el componente si es necesario
        print(f" Vertex AI SDK inicializado para Project: {PROJECT_ID}, Region: {REGION}")

        # --- Parámetros para la ejecución de la pipeline ---
        # Estos pueden ser sobrescritos por variables de entorno o argumentos de CLI si se desea
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
            "kfp_opt_ accelerator_type": KFP_COMPONENT_ACCELERATOR_TYPE, # Nota: nombre con espacio, corregir si es typo
            "kfp_opt_accelerator_count": KFP_COMPONENT_ACCELERATOR_COUNT,
            "polygon_api_key_secret_name_param": run_polygon_secret_name,
            "polygon_api_key_secret_version_param": "latest", # Generalmente 'latest' es suficiente
        }
        # Corregir el typo en el nombre del parámetro si existe
        if "kfp_opt_ accelerator_type" in pipeline_parameter_values:
            pipeline_parameter_values["kfp_opt_accelerator_type"] = pipeline_parameter_values.pop("kfp_opt_ accelerator_type")


        job_display_name = f"algo-trading-v2-{run_pair}-{run_timeframe}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        print(f" Display Name para PipelineJob: {job_display_name}")
        print(f" Template Path (archivo JSON compilado): {pipeline_filename}")
        print(f" Pipeline Root: {PIPELINE_ROOT}")
        print(" Pipeline Parameter Values:")
        print(json.dumps(pipeline_parameter_values, indent=2))

        # Crear el PipelineJob
        job = aiplatform_sdk.PipelineJob(
            display_name=job_display_name,
            template_path=pipeline_filename, # Ruta al archivo JSON compilado
            pipeline_root=PIPELINE_ROOT, # Raíz para artefactos de pipeline
            location=REGION, # Asegúrate que sea la misma región que el bucket de pipeline_root
            enable_caching=False, # Deshabilitar caché para asegurar que todo se ejecute
            parameter_values=pipeline_parameter_values,
        )

        job_resource_name = None # Para guardar el nombre del recurso del job

        try:
            print(f"\nEnviando PipelineJob '{job_display_name}' y esperando su finalización (ejecución síncrona)...")
            # job.submit() # Para ejecución asíncrona
            job.run() # Para ejecución síncrona (bloquea hasta que termina)
            job_resource_name = job.resource_name # Guardar el nombre del recurso después de run() o submit()

            print(f"\n✅ PipelineJob completado.")
            print(f" Job Resource Name: {job.resource_name}") # job.resource_name es el ID completo
            if hasattr(job, 'state') and job.state:
                 print(f" Estado final del Job: {job.state}") # e.g., PipelineState.PIPELINE_STATE_SUCCEEDED
            if job.error: # job.error contendrá el error si el pipeline falló
                print(f" Job Error (si existe): {job.error}")

        except Exception as e:
            print(f"\n❌ Error durante la sumisión o ejecución del PipelineJob:")
            print(f"  Tipo de Excepción: {type(e).__name__}")
            print(f"  Mensaje: {str(e)}")
            print("  Traceback:")
            traceback.print_exc() # Imprime el traceback completo

            # Intentar obtener más detalles del job si ya se creó el objeto
            if job_resource_name: # Si job.submit() o job.run() se alcanzó pero luego falló
                 print(f"  Job Resource Name (si se obtuvo antes del error): {job_resource_name}")
                 # Refrescar el estado del job si es posible (puede que no esté disponible si falló muy temprano)
                 try:
                     job.refresh() # Intenta obtener el último estado del job desde la API
                     current_job_state = job.state if hasattr(job, 'state') else "No disponible"
                     current_job_error = job.error if hasattr(job, 'error') else "No disponible"
                     print(f"  Estado actual del Job (refrescado, si disponible): {current_job_state}")
                     print(f"  Error del Job (refrescado, si disponible): {current_job_error}")
                 except Exception as refresh_err:
                     print(f"  No se pudo refrescar el estado del job: {refresh_err}")
            else:
                print("  No se pudo obtener el Job Resource Name (falló la sumisión inicial o la creación del objeto Job).")
        finally:
            if job_resource_name: # Si tenemos un resource name, podemos construir el enlace
                job_id_for_link = job_resource_name.split('/')[-1] # El ID corto al final del resource name
                pipeline_console_link = f"https://console.cloud.google.com/vertex-ai/pipelines/runs/{job_id_for_link}?project={PROJECT_ID}"
                print(f"\n🔗 Visualizar en Vertex AI Pipelines: {pipeline_console_link}")
            else:
                print("\nℹ️ No se generó un enlace de ejecución ya que el PipelineJob no se pudo someter o falló muy temprano.")
    else:
        print(f"Pipeline compilado a {pipeline_filename}. Para ejecutar, sube este archivo a Vertex AI Pipelines o establece SUBMIT_PIPELINE_TO_VERTEX=true.")