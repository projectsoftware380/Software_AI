# src/components/model_promotion/task.py
"""
Tarea del componente de Promoción de Modelos a Producción.

Responsabilidades:
1.  Cargar las métricas del nuevo modelo y del modelo en producción.
2.  Comparar las métricas según un criterio predefinido (ej: Sharpe Ratio).
3.  Si el nuevo modelo es mejor, copiar sus artefactos (LSTM, filtro) al
    directorio de producción, reemplazando la versión anterior.
4.  Notificar si el modelo fue promovido o no.
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

from src.shared import gcs_utils, constants

# --- Configuración (Sin Cambios) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Lógica Principal de la Tarea (Ajustada) ---
def run_model_promotion(
    new_metrics_dir: str,
    new_lstm_artifacts_dir: str,
    new_filter_model_path: str,
    pair: str,              # <-- AJUSTE: Recibe el par
    timeframe: str,         # <-- AJUSTE: Recibe el timeframe
    production_base_dir: str,
):
    """
    Orquesta el proceso completo de promoción de modelos para un par específico.
    """
    logger.info(f"--- Iniciando promoción de modelo para el par: {pair} ---")

    try:
        # Cargar métricas del nuevo modelo
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_new_metrics_path = gcs_utils.download_gcs_file(f"{new_metrics_dir}/metrics.json", tmp_path)
            with open(local_new_metrics_path) as f:
                new_metrics = json.load(f)

        # Cargar métricas del modelo en producción (si existe)
        prod_metrics = {"sharpe_ratio": -1.0} # Valor por defecto si no hay modelo en producción
        prod_metrics_path = f"{production_base_dir}/{pair}/{timeframe}/metrics.json"
        if gcs_utils.gcs_path_exists(prod_metrics_path):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                local_prod_metrics_path = gcs_utils.download_gcs_file(prod_metrics_path, tmp_path)
                with open(local_prod_metrics_path) as f:
                    prod_metrics = json.load(f)
        else:
            logger.warning(f"No se encontró un modelo en producción para {pair}. El nuevo modelo será promovido automáticamente.")

        # Comparar y decidir la promoción
        new_sharpe = new_metrics.get("sharpe_ratio", 0)
        prod_sharpe = prod_metrics.get("sharpe_ratio", 0)
        
        logger.info(f"Comparando métricas: Nuevo Sharpe ({new_sharpe:.4f}) vs. Producción Sharpe ({prod_sharpe:.4f})")
        
        if new_sharpe > prod_sharpe:
            logger.info(f"✅ ¡Promoción aprobada para {pair}! Copiando artefactos a producción...")
            
            prod_dir = f"{production_base_dir}/{pair}/{timeframe}"
            
            # Copiar artefactos del modelo LSTM
            gcs_utils.copy_gcs_directory(new_lstm_artifacts_dir, f"{prod_dir}/lstm_model")
            
            # Copiar modelo de filtro
            gcs_utils.copy_gcs_blob(new_filter_model_path, f"{prod_dir}/filter_model.pkl")

            # Copiar las nuevas métricas a producción
            gcs_utils.copy_gcs_blob(f"{new_metrics_dir}/metrics.json", f"{prod_dir}/metrics.json")
            
            logger.info(f"Artefactos para {pair} promovidos exitosamente a {prod_dir}")
        else:
            logger.info(f"❌ Promoción rechazada para {pair}. El modelo de producción actual es superior o igual.")

    except Exception as e:
        logger.critical(f"❌ Fallo crítico durante el proceso de promoción para {pair}: {e}", exc_info=True)
        raise

# --- Punto de Entrada para Ejecución como Script (Ajustado) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decide si un nuevo modelo se promueve a producción.")
    
    parser.add_argument("--new-metrics-dir", required=True)
    parser.add_argument("--new-lstm-artifacts-dir", required=True)
    parser.add_argument("--new-filter-model-path", required=True)
    
    # --- AJUSTE AÑADIDO ---
    # Se añaden los argumentos requeridos para que el componente sepa su contexto.
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    
    parser.add_argument("--production-base-dir", required=True)
    
    args = parser.parse_args()
    
    run_model_promotion(
        new_metrics_dir=args.new_metrics_dir,
        new_lstm_artifacts_dir=args.new_lstm_artifacts_dir,
        new_filter_model_path=args.new_filter_model_path,
        pair=args.pair,
        timeframe=args.timeframe,
        production_base_dir=args.production_base_dir,
    )