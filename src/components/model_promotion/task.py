# src/components/model_promotion/task.py
"""
Tarea del componente de decisión de promoción de modelos.

Responsabilidades:
1.  Cargar las métricas del backtest del nuevo modelo candidato.
2.  Cargar las métricas del modelo actualmente en producción (si existe).
3.  Aplicar un conjunto de reglas de veto (ej. drawdown máximo) para rechazar
    modelos inaceptablemente riesgosos.
4.  Calcular un 'score' de promoción ponderado basado en múltiples métricas
    (Sharpe, Profit Factor, Win Rate, etc.).
5.  Comparar el score con un umbral de promoción global.
6.  Si el modelo es promovido:
    - Copiar los artefactos del nuevo modelo (LSTM y PPO) al directorio de
      producción en GCS.
    - Copiar el archivo de métricas del nuevo modelo para que se convierta en
      las nuevas métricas de producción.
7.  Devolver un booleano (`True`/`False`) indicando si el modelo fue promovido.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

# Importar los módulos compartidos
from src.shared import constants, gcs_utils

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# --- Umbrales y Pesos para la Decisión de Promoción ---
# Estos valores podrían externalizarse a un archivo de configuración en el futuro.
MIN_TRADES_FOR_PROMOTION = 100
MIN_PROFIT_FACTOR_ABSOLUTE = 1.0
MIN_SHARPE_ABSOLUTE_FIRST_DEPLOY = 0.5

# Objetivos para normalizar las métricas
TARGETS = {
    "sharpe": 1.5,
    "pf": 1.75,
    "winrate": 0.55,
    "expectancy": 0.05,
    "trades": 200,
}

# Pesos para calcular el score final
WEIGHTS = {
    "sharpe": 0.35,
    "dd": 0.30,
    "pf": 0.25,
    "win": 0.05,
    "exp": 0.03,
    "ntrades": 0.02,
}

# Reglas de Veto
MAX_DRAWDOWN_ABS_THRESHOLD = 25.0  # DD en pips
MAX_DRAWDOWN_REL_TOLERANCE_FACTOR = 1.2  # Nuevo DD no puede ser > 1.2x del de producción

GLOBAL_PROMOTION_SCORE_THRESHOLD = 0.75  # Score mínimo para promover


# --- Funciones de Lógica de Decisión ---

def _load_metrics(gcs_path: str, is_production: bool = False) -> dict:
    """Carga un archivo de métricas desde GCS, con manejo especial para producción."""
    try:
        local_path = gcs_utils.ensure_gcs_path_and_get_local(gcs_path)
        with open(local_path) as f:
            data = json.load(f)
        # Priorizar métricas 'filtered' si existen, de lo contrario usar 'base' o el objeto raíz
        return data.get("filtered", data.get("base", data))
    except Exception as e:
        if is_production:
            logger.warning(f"No se encontró el archivo de métricas de producción en {gcs_path}. "
                           "Se asumirá que es el primer despliegue.")
            return { # Retorna un diccionario con valores "malos" para la comparación
                "trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": -1e9,
                "net_pips": -1e9, "sharpe": -1e9, "max_drawdown": 0.0
            }
        logger.error(f"Error crítico cargando métricas desde {gcs_path}: {e}")
        raise


def _check_veto_conditions(new_mx: dict, prod_mx: dict) -> list[str]:
    """Revisa si el nuevo modelo debe ser vetado por reglas absolutas."""
    veto_reasons = []
    new_dd = abs(new_mx.get("max_drawdown", 1e9))
    prod_dd = abs(prod_mx.get("max_drawdown", 0.0))

    if new_mx.get("trades", 0) < MIN_TRADES_FOR_PROMOTION:
        veto_reasons.append(f"Número de trades ({new_mx.get('trades', 0)}) es menor que el mínimo ({MIN_TRADES_FOR_PROMOTION}).")

    if new_dd > MAX_DRAWDOWN_ABS_THRESHOLD:
        veto_reasons.append(f"Max Drawdown ({new_dd:.2f}%) excede el umbral absoluto ({MAX_DRAWDOWN_ABS_THRESHOLD:.2f}%).")

    if prod_dd > 0 and new_dd > prod_dd * MAX_DRAWDOWN_REL_TOLERANCE_FACTOR:
        veto_reasons.append(f"Max Drawdown ({new_dd:.2f}%) es > {MAX_DRAWDOWN_REL_TOLERANCE_FACTOR:.1f}x el de producción ({prod_dd:.2f}%).")
        
    return veto_reasons


def _calculate_total_score(new_mx: dict, prod_mx: dict) -> float:
    """Calcula el score de promoción ponderado."""
    clip = lambda v: max(0.0, min(v, 1.0)) # Función para normalizar scores a [0, 1]

    sharpe_score = clip(new_mx.get("sharpe", 0.0) / TARGETS["sharpe"])
    pf_score = clip(new_mx.get("profit_factor", 0.0) / TARGETS["pf"])
    win_score = clip(new_mx.get("win_rate", 0.0) / TARGETS["winrate"])
    exp_score = clip(new_mx.get("expectancy", 0.0) / TARGETS["expectancy"])
    trades_score = clip(new_mx.get("trades", 0) / TARGETS["trades"])

    # Score de drawdown: 1.0 si es mejor, 0.0 si es peor.
    new_dd = abs(new_mx.get("max_drawdown", 1e9))
    prod_dd = abs(prod_mx.get("max_drawdown", 1e-9))
    dd_score = clip(1.0 - (new_dd - prod_dd) / prod_dd) if prod_dd > 0 else 1.0

    total_score = (
        sharpe_score * WEIGHTS["sharpe"] +
        dd_score * WEIGHTS["dd"] +
        pf_score * WEIGHTS["pf"] +
        win_score * WEIGHTS["win"] +
        exp_score * WEIGHTS["exp"] +
        trades_score * WEIGHTS["ntrades"]
    )
    
    logger.info(f"Scores individuales: Sharpe={sharpe_score:.2f}, DD={dd_score:.2f}, PF={pf_score:.2f}, Win={win_score:.2f}")
    logger.info(f"Score Total Ponderado: {total_score:.3f}")
    return total_score

# --- Orquestación Principal de la Tarea ---
def run_promotion_decision(
    new_metrics_dir: str,
    new_lstm_artifacts_dir: str,
    new_rl_model_path: str,
    production_base_dir: str,
    pair: str,
    timeframe: str,
) -> bool:
    """
    Orquesta el proceso completo de decisión y promoción.
    Retorna True si el modelo es promovido, False en caso contrario.
    """
    try:
        # 1. Cargar métricas
        new_metrics_path = f"{new_metrics_dir}/metrics.json"
        prod_metrics_path = f"{production_base_dir}/{pair}/{timeframe}/metrics_production.json"
        
        new_metrics = _load_metrics(new_metrics_path)
        prod_metrics = _load_metrics(prod_metrics_path, is_production=True)

        # 2. Verificar vetos
        veto_reasons = _check_veto_conditions(new_metrics, prod_metrics)
        if veto_reasons:
            logger.warning("🚫 Modelo VETADO. Razones:")
            for reason in veto_reasons:
                logger.warning(f"  - {reason}")
            return False

        # 3. Calcular score y decidir
        score = _calculate_total_score(new_metrics, prod_metrics)
        if score < GLOBAL_PROMOTION_SCORE_THRESHOLD:
            logger.info(f"📉 Modelo NO promovido. Score ({score:.3f}) es menor que el umbral ({GLOBAL_PROMOTION_SCORE_THRESHOLD}).")
            return False

        # 4. Promover artefactos
        logger.info(f"✅ Modelo APROBADO para promoción con score {score:.3f}. Copiando artefactos...")
        prod_target_dir = f"{production_base_dir}/{pair}/{timeframe}"

        # Copiar artefactos LSTM
        for artefact in ["model.h5", "scaler.pkl", "params.json"]:
            source = f"{new_lstm_artifacts_dir}/{artefact}"
            destination = f"{prod_target_dir}/{artefact}"
            gcs_utils.copy_gcs_object(source, destination)
            
        # Copiar modelo RL
        gcs_utils.copy_gcs_object(new_rl_model_path, f"{prod_target_dir}/ppo_filter_model.zip")
        
        # Copiar las nuevas métricas como las de producción
        gcs_utils.copy_gcs_object(new_metrics_path, f"{prod_target_dir}/metrics_production.json")
        
        logger.info("🎉 ¡Promoción completada! El nuevo modelo está en producción.")
        return True

    except Exception as e:
        logger.critical(f"❌ Fallo crítico en la decisión de promoción: {e}", exc_info=True)
        raise

# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Decisión de Promoción de Modelos.")
    
    parser.add_argument("--new-metrics-dir", required=True)
    parser.add_argument("--new-lstm-artifacts-dir", required=True)
    parser.add_argument("--new-rl-model-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--production-base-dir", default=constants.PRODUCTION_MODELS_PATH)
    
    args = parser.parse_args()

    promoted = run_promotion_decision(
        new_metrics_dir=args.new_metrics_dir,
        new_lstm_artifacts_dir=args.new_lstm_artifacts_dir,
        new_rl_model_path=args.new_rl_model_path,
        production_base_dir=args.production_base_dir,
        pair=args.pair,
        timeframe=args.timeframe,
    )
    
    # Imprimir el resultado booleano para que KFP lo capture.
    # KFP interpreta 'true' y 'false' (en minúsculas) para las condiciones.
    print(str(promoted).lower())