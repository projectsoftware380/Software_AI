# src/shared/constants.py
"""
Módulo para centralizar todas las constantes y configuraciones del proyecto.

Este archivo sirve como la "fuente de la verdad" para parámetros que se usan
en múltiples lugares, como IDs de proyecto, nombres de buckets, rutas base,
configuraciones de modelos y URIs de imágenes Docker.

Al centralizar la configuración aquí, se facilita la gestión de diferentes
entornos (desarrollo, producción) y se reduce el riesgo de errores por
tener valores harcoded y duplicados en el código.
"""

from __future__ import annotations

# --- Configuración Global del Proyecto GCP ---
PROJECT_ID = "trading-ai-460823"
REGION = "us-central1"
GCS_BUCKET_NAME = "trading-ai-models-460823"

# --- Service Account para Vertex AI Custom Jobs ---
# Usado por el componente que lanza el entrenamiento en Vertex AI.
VERTEX_LSTM_SERVICE_ACCOUNT = "data-ingestion-agent@trading-ai-460823.iam.gserviceaccount.com"

# --- Raíz de la Pipeline en GCS ---
# Directorio base donde KFP/Vertex AI almacenará los artefactos de la pipeline.
PIPELINE_ROOT = f"gs://{GCS_BUCKET_NAME}/pipeline_root_v3" # v3 para la nueva estructura modular

# --- Rutas Base para Artefactos en GCS ---
# Se usan para construir las rutas de salida de cada componente.
BASE_GCS_PATH = f"gs://{GCS_BUCKET_NAME}"
DATA_PATH = f"{BASE_GCS_PATH}/data"
MODELS_PATH = f"{BASE_GCS_PATH}/models"
STAGING_PATH = f"{BASE_GCS_PATH}/staging_for_custom_jobs"
TENSORBOARD_LOGS_PATH = f"{BASE_GCS_PATH}/tensorboard_logs_v3"
BACKTEST_RESULTS_PATH = f"{BASE_GCS_PATH}/backtest_results_v3"
PARAMS_PATH = f"{BASE_GCS_PATH}/params"

# Rutas específicas para cada tipo de artefacto
DATA_FILTERED_FOR_OPT_PATH = f"{PARAMS_PATH}/data_filtered_for_opt_v3"
LSTM_PARAMS_PATH = f"{PARAMS_PATH}/LSTM_v3"
RL_DATA_INPUTS_PATH = f"{PARAMS_PATH}/rl_inputs_v3"

# Rutas de modelos
LSTM_MODELS_PATH = f"{MODELS_PATH}/LSTM_v3"
RL_MODELS_PATH = f"{MODELS_PATH}/RL_v3"
PRODUCTION_MODELS_PATH = f"{MODELS_PATH}/production_v3"


# --- URIs de Imágenes Docker ---
# Imagen base para la mayoría de los componentes KFP que ejecutan scripts locales.
KFP_COMPONENTS_IMAGE_URI = (
    f"europe-west1-docker.pkg.dev/{PROJECT_ID}/data-ingestion-repo/data-ingestion-agent:latest"
)

# Imagen "runner" para el entrenamiento del modelo LSTM en un Vertex AI Custom Job.
VERTEX_LSTM_TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/trading-images/runner-lstm:latest"
)

# --- Configuración de Máquinas y Aceleradores para Vertex AI ---
# Valores por defecto para el Custom Job de entrenamiento del LSTM.
DEFAULT_VERTEX_LSTM_MACHINE_TYPE = "n1-standard-4"
DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT = 0

# --- Configuración de Hiperparámetros por Defecto de la Pipeline ---
DEFAULT_N_TRIALS = 2
DEFAULT_PAIR = "EURUSD"
DEFAULT_TIMEFRAME = "15minute"

# --- Configuración de Secret Manager ---
# Nombre del secreto que contiene la API Key de Polygon.
POLYGON_API_KEY_SECRET_NAME = "polygon-api-key"
POLYGON_API_KEY_SECRET_VERSION = "latest"

# --- Configuración para Pub/Sub Notificaciones ---
SUCCESS_TOPIC_ID = "data-ingestion-success"
FAILURE_TOPIC_ID = "data-ingestion-failures"