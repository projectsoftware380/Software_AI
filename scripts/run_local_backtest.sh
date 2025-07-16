#!/bin/bash
# ------------------------------------------------------------------
# run_local_backtest.sh: Ejecuta el backtesting localmente.
# ------------------------------------------------------------------
#
# Este script invoca el componente de backtesting. Descargará el
# modelo LSTM y el modelo de filtro entrenados desde GCS, junto con
# los datos de holdout, y ejecutará el backtest en la máquina local.
# Los resultados (métricas) se guardarán en GCS.
#
# Requisitos:
# - Haber ejecutado la pipeline de entrenamiento en GCP.
# - Tener Google Cloud SDK autenticado.

set -e

# --- Configuración ---
export PROJECT_ID="$(gcloud config get-value project)"
export PAIR="EURUSD"
export TIMEFRAME="m1"

# --- Rutas de GCS ---
# Rutas donde se guardaron los modelos entrenados y los datos de holdout.
export LSTM_MODEL_DIR="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/models/lstm/trained/${PAIR}/${TIMEFRAME}"
export FILTER_MODEL_PATH="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/models/filter/trained/${PAIR}/${TIMEFRAME}/filter_model.joblib"
export HOLDOUT_DATA_PATH="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/data/prepared/${PAIR}/${TIMEFRAME}/holdout/holdout_data.parquet"

# Ruta de salida para las métricas del backtest.
export OUTPUT_GCS_DIR="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/backtest_results/${PAIR}/${TIMEFRAME}"

# --- Ejecución del Componente ---
echo "🚀 Iniciando el backtesting local para ${PAIR}..."
echo "   - Modelo LSTM desde: ${LSTM_MODEL_DIR}"
echo "   - Modelo de filtro desde: ${FILTER_MODEL_PATH}"
echo "   - Datos de holdout desde: ${HOLDOUT_DATA_PATH}"

python -m src.components.backtest.task \
    --lstm-model-dir "${LSTM_MODEL_DIR}" \
    --filter-model-path "${FILTER_MODEL_PATH}" \
    --features-path "${HOLDOUT_DATA_PATH}" \
    --output-gcs-dir "${OUTPUT_GCS_DIR}" \
    --pair "${PAIR}" \
    --timeframe "${TIMEFRAME}"

echo "✅ Backtesting completado. Las métricas están en GCS."
