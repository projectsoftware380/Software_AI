"""
Fuente única de configuración para todo el proyecto:
– IDs y regiones de GCP
– rutas de GCS
– defaults de Vertex AI
"""

from __future__ import annotations

# ──────────────────── CONFIG GLOBAL GCP ────────────────────
PROJECT_ID: str = "trading-ai-460823"
REGION: str = "europe-west1"          # Coincide con Artifact Registry
GCS_BUCKET_NAME: str = "trading-ai-models-460823"

# ──────────────────── SERVICE ACCOUNTS ─────────────────────
VERTEX_LSTM_SERVICE_ACCOUNT: str = (
    "data-ingestion-agent@trading-ai-460823.iam.gserviceaccount.com"
)

# ──────────────────── RUTAS GCS BASE ───────────────────────
PIPELINE_ROOT = f"gs://{GCS_BUCKET_NAME}/pipeline_root_v3"
BASE_GCS_PATH = f"gs://{GCS_BUCKET_NAME}"

DATA_PATH = f"{BASE_GCS_PATH}/data"
MODELS_PATH = f"{BASE_GCS_PATH}/models"
STAGING_PATH = f"{BASE_GCS_PATH}/staging_for_custom_jobs"
TENSORBOARD_LOGS_PATH = f"{BASE_GCS_PATH}/tensorboard_logs_v3"
BACKTEST_RESULTS_PATH = f"{BASE_GCS_PATH}/backtest_results_v3"
PARAMS_PATH = f"{BASE_GCS_PATH}/params"

# Rutas específicas
DATA_FILTERED_FOR_OPT_PATH = f"{PARAMS_PATH}/data_filtered_for_opt_v3"
LSTM_PARAMS_PATH            = f"{PARAMS_PATH}/LSTM_v3"
RL_DATA_INPUTS_PATH         = f"{PARAMS_PATH}/rl_inputs_v3"

# Rutas de modelos
LSTM_MODELS_PATH      = f"{MODELS_PATH}/LSTM_v3"
RL_MODELS_PATH        = f"{MODELS_PATH}/RL_v3"
PRODUCTION_MODELS_PATH = f"{MODELS_PATH}/production_v3"

# ==============================================================================
# === AJUSTE CLAVE: Se elimina la URI completa y específica de la imagen. ===
# El script `run_pipeline.ps1` y `main.py` ahora se encargan de construir
# y pasar la URI completa y versionada dinámicamente.
#
# Ya no necesitamos una constante para la URI de la imagen aquí.
# Si necesitaras las partes base para construir la URI en otro lugar,
# podrías definirlas así:
DOCKER_REPO_NAME: str = "data-ingestion-repo"
DOCKER_IMAGE_NAME: str = "data-ingestion-agent"
# ==============================================================================

# ──────────────────── VERTEX AI DEFAULTS ───────────────────
DEFAULT_VERTEX_LSTM_MACHINE_TYPE     = "n1-standard-4"
DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT = 0

# ──────────────────── PIPELINE DEFAULTS ────────────────────
DEFAULT_N_TRIALS   = 2
DEFAULT_PAIR       = "EURUSD"
DEFAULT_TIMEFRAME  = "15minute"

# ──────────────────── SECRET MANAGER ───────────────────────
POLYGON_API_KEY_SECRET_NAME    = "polygon-api-key"
POLYGON_API_KEY_SECRET_VERSION = "latest"

# ──────────────────── PUB/SUB TOPICS ───────────────────────
SUCCESS_TOPIC_ID = "data-ingestion-success"
FAILURE_TOPIC_ID = "data-ingestion-failures"