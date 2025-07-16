#!/bin/bash
# ---------------------------------------------------------------------
# validate_hybrid_flow.sh: Orquesta y valida el flujo híbrido completo.
# ---------------------------------------------------------------------
#
# Este script ejecuta todos los pasos del flujo híbrido en secuencia:
# 1. Ingesta de datos (Local)
# 2. Preparación de datos (Local)
# 3. Entrenamiento de modelos (GCP, síncrono)
# 4. Backtesting (Local)
# 5. Promoción de modelos (Local)
#
# El script se detendrá si alguno de los pasos falla.

set -e

echo "======================================================"
echo "🚀 INICIANDO VALIDACIÓN DEL FLUJO HÍBRIDO COMPLETO"
echo "======================================================"

# Paso 1: Ingesta de datos local
echo "
--- PASO 1: Ejecutando ingesta de datos local... ---"
bash ./scripts/run_local_ingestion.sh
echo "--- PASO 1: Ingesta de datos completada ---"

# Paso 2: Preparación de datos local
echo "
--- PASO 2: Ejecutando preparación de datos local... ---"
bash ./scripts/run_local_preparation.sh
echo "--- PASO 2: Preparación de datos completada ---"

# Paso 3: Entrenamiento en GCP (síncrono)
echo "
--- PASO 3: Lanzando pipeline de entrenamiento en GCP (esperando finalización)... ---"
python ./scripts/run_gcp_training.py
echo "--- PASO 3: Pipeline de entrenamiento en GCP completada ---"

# Paso 4: Backtesting local
echo "
--- PASO 4: Ejecutando backtesting local... ---"
bash ./scripts/run_local_backtest.sh
echo "--- PASO 4: Backtesting local completado ---"

# Paso 5: Promoción de modelos local
echo "
--- PASO 5: Ejecutando promoción de modelos local... ---"
bash ./scripts/run_local_promotion.sh
echo "--- PASO 5: Promoción de modelos completada ---"

echo "
======================================================"
echo "✅ VALIDACIÓN DEL FLUJO HÍBRIDO COMPLETADA EXITOSAMENTE"
echo "======================================================"
