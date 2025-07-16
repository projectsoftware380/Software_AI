# src/components/model_promotion/task.py
"""
Tarea del componente de Promoción de Modelos a Producción. (Versión con Logging Robusto)

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

# Módulos internos
from src.shared import gcs_utils, constants

# --- Configuración del Logging ---
from src.shared.logging_config import setup_logging

setup_logging() # Asegurar que el logging esté configurado
logger = logging.getLogger(__name__)

# --- Lógica Principal de la Tarea ---
def run_model_promotion(
    new_metrics_dir: str,
    new_lstm_artifacts_dir: str,
    new_filter_model_path: str,
    pair: str,
    timeframe: str,
    production_base_dir: str,
):
    """
    Orquesta el proceso completo de promoción de modelos para un par específico.
    """
    # [LOG] Punto de control inicial con todos los parámetros recibidos.
    logger.info(f"▶️ Iniciando model_promotion para el par '{pair}':")
    logger.info(f"  - Directorio de Métricas Nuevas: {new_metrics_dir}")
    logger.info(f"  - Directorio de Artefactos LSTM Nuevos: {new_lstm_artifacts_dir}")
    logger.info(f"  - Ruta del Modelo Filtro Nuevo: {new_filter_model_path}")
    logger.info(f"  - Directorio Base de Producción: {production_base_dir}")

    try:
        # Cargar métricas del nuevo modelo
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            logger.info(f"Descargando métricas del nuevo modelo desde: {new_metrics_dir}/metrics.json")
            local_new_metrics_path = gcs_utils.download_gcs_file(f"{new_metrics_dir}/metrics.json", tmp_path)
            if local_new_metrics_path is None:
                raise FileNotFoundError(f"No se encontró el archivo de métricas del nuevo modelo en {new_metrics_dir}")
                
            with open(local_new_metrics_path) as f:
                new_metrics = json.load(f)
            logger.info(f"✅ Métricas del nuevo modelo cargadas: {new_metrics}")

        # Cargar métricas del modelo en producción (si existe)
        prod_metrics = {"sharpe_ratio": -999.0} # Valor por defecto muy bajo si no hay modelo en producción
        prod_metrics_path = f"{production_base_dir}/{pair}/{timeframe}/metrics.json"
        
        logger.info(f"Buscando modelo en producción en: {prod_metrics_path}")
        if gcs_utils.gcs_path_exists(prod_metrics_path):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                local_prod_metrics_path = gcs_utils.download_gcs_file(prod_metrics_path, tmp_path)
                with open(local_prod_metrics_path) as f:
                    prod_metrics = json.load(f)
            logger.info(f"✅ Métricas del modelo en producción cargadas: {prod_metrics}")
        else:
            logger.warning(f"⚠️ No se encontró un modelo en producción para {pair}. El nuevo modelo será promovido automáticamente si cumple un umbral mínimo.")

        # Comparar y decidir la promoción
        new_sharpe = new_metrics.get("sharpe_ratio", 0.0)
        prod_sharpe = prod_metrics.get("sharpe_ratio", -999.0)
        
        # [LOG] Registro de la comparación que se va a realizar.
        logger.info(f"--- Decisión de Promoción ---")
        logger.info(f"  - Sharpe Ratio del Nuevo Modelo: {new_sharpe:.4f}")
        logger.info(f"  - Sharpe Ratio del Modelo en Producción: {prod_sharpe:.4f}")
        
        if new_sharpe > prod_sharpe:
            logger.info(f"✅ ¡Promoción APROBADA para {pair}! Copiando artefactos a producción...")
            
            prod_dir = f"{production_base_dir}/{pair}/{timeframe}"
            
            # Copiar artefactos del modelo LSTM
            logger.info(f"Copiando directorio LSTM: {new_lstm_artifacts_dir} -> {prod_dir}/lstm_model")
            gcs_utils.copy_gcs_directory(new_lstm_artifacts_dir, f"{prod_dir}/lstm_model")
            
            # Copiar modelo de filtro
            logger.info(f"Copiando modelo filtro: {new_filter_model_path} -> {prod_dir}/filter_model.pkl")
            gcs_utils.copy_gcs_blob(new_filter_model_path, f"{prod_dir}/filter_model.pkl")

            # Copiar las nuevas métricas a producción
            logger.info(f"Copiando archivo de métricas a: {prod_dir}/metrics.json")
            gcs_utils.copy_gcs_blob(f"{new_metrics_dir}/metrics.json", f"{prod_dir}/metrics.json")
            
            logger.info(f"🎉 Artefactos para {pair} promovidos exitosamente a {prod_dir}")
        else:
            logger.info(f"❌ Promoción RECHAZADA para {pair}. El modelo de producción actual es superior o igual.")

    except Exception as e:
        logger.exception(f"❌ Fallo fatal durante el proceso de promoción para '{pair}'. Error: {e}")
        raise
        
    logger.info(f"🏁 Componente model_promotion para '{pair}' completado exitosamente.")

# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decide si un nuevo modelo se promueve a producción.")
    
    parser.add_argument("--new-metrics-dir", required=True)
    parser.add_argument("--new-lstm-artifacts-dir", required=True)
    parser.add_argument("--new-filter-model-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--production-base-dir", required=True)
    
    args = parser.parse_args()

    # [LOG] Registro de los argumentos recibidos.
    logger.info("Componente 'model_promotion' iniciado con los siguientes argumentos:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
        
    run_model_promotion(
        new_metrics_dir=args.new_metrics_dir,
        new_lstm_artifacts_dir=args.new_lstm_artifacts_dir,
        new_filter_model_path=args.new_filter_model_path,
        pair=args.pair,
        timeframe=args.timeframe,
        production_base_dir=args.production_base_dir,
    )