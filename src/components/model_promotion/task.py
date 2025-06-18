# src/components/model_promotion/task.py
"""
Decide si promover el nuevo modelo a producción usando una lógica de 3 pasos.
Escribe «true» o «false» en /tmp/model_promoted.txt para que KFP lo consuma.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.shared import constants, gcs_utils

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Umbrales de Promoción ---
MIN_SHARPE_FOR_PROMOTION = 0.5
MIN_IMPROVEMENT_FACTOR = 1.25 # El Sharpe filtrado debe ser al menos un 25% mejor que el base
MIN_TRADES_FOR_PROMOTION = 50

# --- Helpers ---

def _load_metrics(path: str) -> dict | None:
    """Carga un archivo de métricas JSON desde GCS. Devuelve None si no existe."""
    try:
        local = gcs_utils.ensure_gcs_path_and_get_local(path)
        with open(local, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        logger.warning("No se encontró el archivo de métricas en %s", path)
        return None

# --- Lógica de Promoción ---

def run_promotion_decision(
    *,
    new_metrics_dir: str,
    new_lstm_artifacts_dir: str,
    new_filter_model_path: str,
    production_base_dir: str,
    pair: str,
    timeframe: str,
) -> bool:
    """
    Ejecuta la lógica de decisión de 3 pasos para promover un modelo.
    """
    # 1. Cargar las métricas necesarias
    new_metrics_path = f"{new_metrics_dir.rstrip('/')}/metrics.json"
    prod_metrics_path = f"{production_base_dir.rstrip('/')}/{pair}/{timeframe}/metrics_production.json"

    new_metrics_data = _load_metrics(new_metrics_path)
    prod_metrics_data = _load_metrics(prod_metrics_path)

    if not new_metrics_data:
        logger.error("🚫 No se pudieron cargar las métricas del nuevo modelo. No se puede promover.")
        return False
        
    base_metrics = new_metrics_data.get("base", {})
    filtered_metrics = new_metrics_data.get("filtered", {})
    prod_metrics = prod_metrics_data.get("filtered", {}) if prod_metrics_data else {}

    # 2. Realizar las 3 validaciones
    
    # Criterio 1: ¿La estrategia filtrada es viable por sí misma?
    if filtered_metrics.get("sharpe", 0) < MIN_SHARPE_FOR_PROMOTION:
        logger.warning(f"🚫 VETO: El Sharpe Ratio de la estrategia filtrada ({filtered_metrics.get('sharpe', 0):.2f}) es menor que el mínimo requerido ({MIN_SHARPE_FOR_PROMOTION}).")
        return False
    if filtered_metrics.get("trades", 0) < MIN_TRADES_FOR_PROMOTION:
        logger.warning(f"🚫 VETO: El número de operaciones filtradas ({filtered_metrics.get('trades', 0)}) es menor que el mínimo requerido ({MIN_TRADES_FOR_PROMOTION}).")
        return False
    logger.info("✅ Criterio 1/3 PASADO: La nueva estrategia filtrada es viable.")

    # Criterio 2: ¿El filtro mejora significativamente la estrategia base?
    base_sharpe = base_metrics.get("sharpe", 0)
    filtered_sharpe = filtered_metrics.get("sharpe", 0)
    
    # Manejar el caso de Sharpe base negativo o cero
    improvement_check = (filtered_sharpe > base_sharpe * MIN_IMPROVEMENT_FACTOR) if base_sharpe > 0 else (filtered_sharpe > 0)
    
    if not improvement_check:
        logger.warning(f"🚫 VETO: El filtro no mejora suficientemente la estrategia base. Sharpe Base: {base_sharpe:.2f}, Sharpe Filtrado: {filtered_sharpe:.2f}.")
        return False
    logger.info("✅ Criterio 2/3 PASADO: El filtro aporta un valor significativo.")

    # Criterio 3: ¿La nueva estrategia filtrada es mejor que la de producción?
    prod_sharpe = prod_metrics.get("sharpe", -1.0) # Si no hay modelo en prod, su sharpe es -1
    if filtered_sharpe <= prod_sharpe:
        logger.warning(f"🚫 VETO: La nueva estrategia filtrada (Sharpe: {filtered_sharpe:.2f}) no supera a la de producción (Sharpe: {prod_sharpe:.2f}).")
        return False
    logger.info("✅ Criterio 3/3 PASADO: La nueva estrategia supera al modelo en producción.")

    # 3. Si se pasan todas las validaciones, copiar los artefactos
    logger.info("🎉 ¡Todos los criterios cumplidos! Promoviendo el nuevo modelo a producción.")
    
    dest_dir = f"{production_base_dir.rstrip('/')}/{pair}/{timeframe}"
    
    # Copiar artefactos LSTM
    for fname in ("model.keras", "scaler.pkl", "params.json"):
        gcs_utils.copy_gcs_object(f"{new_lstm_artifacts_dir}/{fname}", f"{dest_dir}/{fname}")
    
    # Copiar artefactos del Filtro
    for fname in ("filter_model.pkl", "filter_params.json"):
        gcs_utils.copy_gcs_object(f"{new_filter_model_path}/{fname}", f"{dest_dir}/{fname}")
        
    # Copiar el nuevo informe de métricas como el de producción
    gcs_utils.copy_gcs_object(new_metrics_path, f"{dest_dir}/metrics_production.json")

    logger.info("🚀 Modelo promovido con éxito a: %s", dest_dir)
    return True

# --- CLI ---
if __name__ == "__main__":
    p = argparse.ArgumentParser("Promotion Decision Task")
    p.add_argument("--new-metrics-dir", required=True)
    p.add_argument("--new-lstm-artifacts-dir", required=True)
    p.add_argument(
        "--new-filter-model-path",
        default="/tmp/filter_model",
    )
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--production-base-dir", default=constants.PRODUCTION_MODELS_PATH)
    args = p.parse_args()

    promoted = run_promotion_decision(
        new_metrics_dir=args.new_metrics_dir,
        new_lstm_artifacts_dir=args.new_lstm_artifacts_dir,
        new_filter_model_path=args.new_filter_model_path,
        production_base_dir=args.production_base_dir,
        pair=args.pair,
        timeframe=args.timeframe,
    )

    out_file = Path("/tmp/model_promoted.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("true" if promoted else "false")
    print(f"Resultado de la Promoción: {'Sí' if promoted else 'No'}")