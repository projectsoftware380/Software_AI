# src/components/model_promotion/task.py
"""
Decide si promover el nuevo modelo a producción comparando métricas.
Escribe 'true' o 'false' en /tmp/model_promoted.txt para que KFP
lo recoja como salida (fileOutputs).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.shared import constants, gcs_utils
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────── parámetros y pesos ────────────────────────────────────────────
MIN_TRADES_FOR_PROMOTION = 100
MAX_DRAWDOWN_ABS_THRESHOLD = 25.0
MAX_DRAWDOWN_REL_TOLERANCE_FACTOR = 1.2
GLOBAL_PROMOTION_SCORE_THRESHOLD = 0.75

TARGETS = dict(sharpe=1.5, pf=1.75, winrate=0.55, expectancy=0.05, trades=200)
WEIGHTS = dict(sharpe=0.35, dd=0.30, pf=0.25, win=0.05, exp=0.03, ntrades=0.02)

# ─────── utilidades ────────────────────────────────────────────────────
def _load_metrics(path: str, *, production: bool = False) -> dict:
    try:
        local = gcs_utils.ensure_gcs_path_and_get_local(path)
        with open(local) as fh:
            data = json.load(fh)
        return data.get("filtered", data.get("base", data))
    except Exception:
        if production:
            logger.warning("Métricas de producción no encontradas: %s  (primer despliegue)", path)
            return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                    "expectancy": -1e9, "net_pips": -1e9, "sharpe": -1e9,
                    "max_drawdown": 0.0}
        raise

def _check_vetos(new: dict, prod: dict) -> list[str]:
    reasons = []
    new_dd, prod_dd = abs(new.get("max_drawdown", 1e9)), abs(prod.get("max_drawdown", 0))
    if new.get("trades", 0) < MIN_TRADES_FOR_PROMOTION:
        reasons.append("Muy pocos trades")
    if new_dd > MAX_DRAWDOWN_ABS_THRESHOLD:
        reasons.append("Drawdown absoluto excesivo")
    if prod_dd > 0 and new_dd > prod_dd * MAX_DRAWDOWN_REL_TOLERANCE_FACTOR:
        reasons.append("Drawdown > 1.2× producción")
    return reasons

def _score(new: dict, prod: dict) -> float:
    clip = lambda v: max(0.0, min(v, 1.0))
    sharpe = clip(new.get("sharpe", 0) / TARGETS["sharpe"])
    pf     = clip(new.get("profit_factor", 0) / TARGETS["pf"])
    win    = clip(new.get("win_rate", 0) / TARGETS["winrate"])
    exp    = clip(new.get("expectancy", 0) / TARGETS["expectancy"])
    trades = clip(new.get("trades", 0) / TARGETS["trades"])

    new_dd, prod_dd = abs(new.get("max_drawdown", 1e9)), abs(prod.get("max_drawdown", 1e-9))
    dd      = clip(1.0 - (new_dd - prod_dd) / prod_dd) if prod_dd else 1.0

    total = (sharpe * WEIGHTS["sharpe"] + dd * WEIGHTS["dd"] +
             pf * WEIGHTS["pf"] + win * WEIGHTS["win"] +
             exp * WEIGHTS["exp"] + trades * WEIGHTS["ntrades"])
    logger.info("Scores → Sharpe %.2f, DD %.2f, PF %.2f, Win %.2f  => Total %.3f",
                sharpe, dd, pf, win, total)
    return total

# ─────── función principal ────────────────────────────────────────────
def run_promotion_decision(
    *,
    new_metrics_dir: str,
    new_lstm_artifacts_dir: str,
    new_rl_model_path: str,
    production_base_dir: str,
    pair: str,
    timeframe: str,
) -> bool:
    new_metrics_path  = f"{new_metrics_dir}/metrics.json"
    prod_metrics_path = f"{production_base_dir}/{pair}/{timeframe}/metrics_production.json"

    new  = _load_metrics(new_metrics_path)
    prod = _load_metrics(prod_metrics_path, production=True)

    vetos = _check_vetos(new, prod)
    if vetos:
        logger.warning("🚫 Modelo vetado: %s", "; ".join(vetos))
        return False

    if _score(new, prod) < GLOBAL_PROMOTION_SCORE_THRESHOLD:
        logger.info("📉 Score insuficiente para promocionar")
        return False

    # Copiar artefactos
    dest_dir = f"{production_base_dir}/{pair}/{timeframe}"
    for f in ["model.h5", "scaler.pkl", "params.json"]:
        gcs_utils.copy_gcs_object(f"{new_lstm_artifacts_dir}/{f}", f"{dest_dir}/{f}")
    gcs_utils.copy_gcs_object(new_rl_model_path, f"{dest_dir}/ppo_filter_model.zip")
    gcs_utils.copy_gcs_object(new_metrics_path, f"{dest_dir}/metrics_production.json")

    logger.info("🎉 Modelo promovido a producción: %s", dest_dir)
    return True

# ─────── CLI / entrypoint ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Promotion decision task")
    p.add_argument("--new-metrics-dir", required=True)
    p.add_argument("--new-lstm-artifacts-dir", required=True)
    p.add_argument("--new-rl-model-path", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--production-base-dir", default=constants.PRODUCTION_MODELS_PATH)
    args = p.parse_args()

    promoted = run_promotion_decision(
        new_metrics_dir=args.new_metrics_dir,
        new_lstm_artifacts_dir=args.new_lstm_artifacts_dir,
        new_rl_model_path=args.new_rl_model_path,
        production_base_dir=args.production_base_dir,
        pair=args.pair,
        timeframe=args.timeframe,
    )

    # Salida para KFP
    Path("/tmp/model_promoted.txt").write_text("true" if promoted else "false")
    print(f"Promoted: {promoted}")
