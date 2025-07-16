#!/bin/bash
# ------------------------------------------------------------------
# run_local_ingestion.sh: Ejecuta la ingesta de datos localmente.
# ------------------------------------------------------------------
#
# Este script invoca el componente de ingesta de datos para que se
# ejecute en la máquina local. Descargará los datos de Dukascopy
# y los subirá directamente a un bucket de Google Cloud Storage (GCS).
#
# Requisitos:
# - Tener Google Cloud SDK autenticado (`gcloud auth application-default login`).
# - Tener las variables de entorno del proyecto configuradas o definirlas aquí.

set -e # Terminar el script si un comando falla

# --- Configuración ---
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="europe-west1" # O la región que prefieras
export PAIR="EURUSD"
export TIMEFRAME="m1"
export END_DATE="2023-12-31"

# --- Definición de la ruta de salida en GCS ---
# El componente creará un subdirectorio basado en el par y timeframe.
export OUTPUT_GCS_BASE_PATH="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/data/ingested"

# --- Ejecución del Componente ---
echo "🚀 Iniciando la ingesta de datos local para el par ${PAIR}..."
echo "   - Proyecto: ${PROJECT_ID}"
echo "   - Datos se subirán a: ${OUTPUT_GCS_BASE_PATH}/${PAIR}"

# Invocamos el task del componente directamente con Python
python -m src.components.dukascopy_ingestion.task \
    --project-id "${PROJECT_ID}" \
    --pair "${PAIR}" \
    --timeframe "${TIMEFRAME}" \
    --end-date "${END_DATE}" \
    --output-gcs-path "${OUTPUT_GCS_BASE_PATH}"

echo "✅ Ingesta de datos completada. Los archivos están en GCS."
