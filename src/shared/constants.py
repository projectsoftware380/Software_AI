"""
Módulo para centralizar todas las constantes y configuraciones del proyecto.

Este archivo sirve como la "fuente de la verdad" para parámetros que se usan
en múltiples lugares, como IDs de proyecto, nombres de buckets, rutas base,
configuraciones de modelos y URIs de imágenes Docker.

Al centralizar la configuración aquí, se facilita la gestión de diferentes
entornos (desarrollo, producción) y se reduce el riesgo de errores por
tener valores hardcoded y duplicados en el código.
"""

from __future__ import annotations

# --- Configuración Global del Proyecto GCP ---
PROJECT_ID = "trading-ai-460823"
REGION = "europe-west1"  # Corregido para coincidir con Artifact Registry
GCS_BUCKET_NAME = "trading-ai-models-460823"

# --- Service Account para Vertex AI Custom Jobs ---
VERTEX_LSTM_SERVICE_ACCOUNT = (
    "data-ingestion-agent@trading-ai-460823.iam.gserviceaccount.com"
)

# --- Raíz de la Pipeline en GCS ---
PIPELINE_ROOT = f"gs://{GCS_BUCKET_NAME}/pipeline_root_v3"

# --- Rutas Base para Artefactos en GCS ---
BASE_GCS_PATH = f"gs://{GCS_BUCKET_NAME}"
DATA_PATH = f"{BASE_GCS_PATH}/data"
MODELS_PATH = f"{BASE_GCS_PATH}/models"
STAGING_PATH = f"{BASE_GCS_PATH}/staging_for_custom_jobs"
TENSORBOARD_LOGS_PATH = f"{BASE_GCS_PATH}/tensorboard_logs_v3"
BACKTEST_RESULTS_PATH = f"{BASE_GCS_PATH}/backtest_results_v3"
PARAMS_PATH = f"{BASE_GCS_PATH}/params"

# Rutas específicas
DATA_FILTERED_FOR_OPT_PATH = f"{PARAMS_PATH}/data_filtered_for_opt_v3"
LSTM_PARAMS_PATH = f"{PARAMS_PATH}/LSTM_v3"
RL_DATA_INPUTS_PATH = f"{PARAMS_PATH}/rl_inputs_v3"

# Rutas de modelos
LSTM_MODELS_PATH = f"{MODELS_PATH}/LSTM_v3"
RL_MODELS_PATH = f"{MODELS_PATH}/RL_v3"
PRODUCTION_MODELS_PATH = f"{MODELS_PATH}/production_v3"

# --- URIs de Imágenes Docker ---
# Imagen general para componentes KFP
KFP_COMPONENTS_IMAGE_URI = (
    f"europe-west1-docker.pkg.dev/{PROJECT_ID}/data-ingestion-repo/data-ingestion-agent:latest"
)

# Imagen para entrenamiento LSTM en Vertex AI
# CORREGIDA: ahora apunta a europe-west1 y al mismo repositorio
VERTEX_LSTM_TRAINER_IMAGE_URI = (
    f"europe-west1-docker.pkg.dev/{PROJECT_ID}/data-ingestion-repo/data-ingestion-agent:latest"
)

# --- Configuración de Máquinas y Aceleradores para Vertex AI ---
DEFAULT_VERTEX_LSTM_MACHINE_TYPE = "n1-standard-4"
DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT = 0

# --- Hiperparámetros por Defecto ---
DEFAULT_N_TRIALS = 2
DEFAULT_PAIR = "EURUSD"
DEFAULT_TIMEFRAME = "15minute"

# --- Secret Manager ---
POLYGON_API_KEY_SECRET_NAME = "polygon-api-key"
POLYGON_API_KEY_SECRET_VERSION = "latest"

# --- Pub/Sub Notificaciones ---
SUCCESS_TOPIC_ID = "data-ingestion-success"
FAILURE_TOPIC_ID = "data-ingestion-failures"
