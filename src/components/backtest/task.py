# src/components/backtest/task.py
"""
Ejecuta el backtest (LSTM + filtro PPO) y sube resultados a GCS.
Escribe:
    • /tmp/backtest_dir.txt  → carpeta final en GCS (para KFP)
    • /tmp/kfp_metrics.json  → métricas en formato Vertex UI (para KFP)
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from stable_baselines3 import PPO

from src.shared import constants, gcs_utils, indicators

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ATR_LEN = 14
COST_PIPS = 0.8


# ──────────────────────────── helpers  ────────────────────────────
def _load_artifacts(lstm_model_dir: str, rl_model_path: str) -> Tuple[tf.keras.Model, any, dict, PPO]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model = tf.keras.models.load_model(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.h5", tmp), compile=False
        )
        scaler = joblib.load(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp)
        )
        hp = json.loads(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp).read_text()
        )
        ppo = PPO.load(
            gcs_utils.download_gcs_file(rl_model_path, tmp)
        )
    logger.info("Artefactos cargados ✔️")
    return model, scaler, hp, ppo


def _prepare_backtest_data(features_path: str, hp: dict) -> pd.DataFrame:
    df_raw = pd.read_parquet(gcs_utils.ensure_gcs_path_and_get_local(features_path))
    df_ind = indicators.build_indicators(df_raw, hp, atr_len=ATR_LEN, drop_na=True)
    if df_ind.isna().any().any():
        raise ValueError("Persisten NaNs tras calcular indicadores")
    return df_ind.reset_index(drop=True)


# (Las funciones _generate_predictions, _run_backtest_simulation y
#  _calculate_metrics permanecen sin cambios — se omiten aquí por brevedad)
#  … copia tus implementaciones actuales …


# ───────────────────────────  función orquestadora  ──────────────────────────
def run_backtest(
    *,
    lstm_model_dir: str,
    rl_model_path: str,
    features_path: str,
    pair: str,
    timeframe: str,
) -> tuple[str, str]:
    """
    Devuelve:
        output_dir (str)  → carpeta GCS donde quedaron CSV + metrics.json
        kfp_metrics_path  → ruta local al JSON formateado para Vertex UI
    """
    tick = 0.01 if pair.endswith("JPY") else 0.0001
    model, scaler, hp, ppo = _load_artifacts(lstm_model_dir, rl_model_path)
    df_bt = _prepare_backtest_data(features_path, hp)

    up, dn, emb, closes, atr = _generate_predictions(df_bt, model, scaler, hp, tick)
    accept_mask, _ = ppo.predict(np.hstack([np.column_stack([up, dn]), emb]), deterministic=True)

    trades_base = _run_backtest_simulation(up, dn, closes, atr,
                                           np.ones_like(accept_mask), hp, tick, use_filter=False)
    trades_filt = _run_backtest_simulation(up, dn, closes, atr,
                                           accept_mask, hp, tick, use_filter=True)

    metrics = {
        "base": _calculate_metrics(trades_base),
        "filtered": _calculate_metrics(trades_filt),
    }

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir = f"{constants.BACKTEST_RESULTS_PATH}/{pair}/{timeframe}/{ts}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if not trades_base.empty:
            csv_base = tmp / "trades_base.csv"
            trades_base.to_csv(csv_base, index=False)
            gcs_utils.upload_gcs_file(csv_base, f"{output_dir}/trades_base.csv")

        if not trades_filt.empty:
            csv_filt = tmp / "trades_filtered.csv"
            trades_filt.to_csv(csv_filt, index=False)
            gcs_utils.upload_gcs_file(csv_filt, f"{output_dir}/trades_filtered.csv")

        json_metrics = tmp / "metrics.json"
        json_metrics.write_text(json.dumps(metrics, indent=2))
        gcs_utils.upload_gcs_file(json_metrics, f"{output_dir}/metrics.json")

        # Formato Vertex UI
        kfp_json = tmp / "kfp_metrics.json"
        kfp_json.write_text(json.dumps({
            "metrics": [
                {"name": f"filtered-{k}", "numberValue": (v if np.isfinite(v) else 0), "format": "RAW"}
                for k, v in metrics["filtered"].items()
            ]
        }))

    logger.info("Backtest finalizado ✔️  Resultados: %s", output_dir)
    return output_dir, str(kfp_json)


# ────────────────────────────  CLI (entrypoint)  ────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Backtest task")
    p.add_argument("--lstm-model-dir", required=True)
    p.add_argument("--rl-model-path", required=True)
    p.add_argument("--features-path", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    args = p.parse_args()

    out_dir, kfp_metrics_file = run_backtest(
        lstm_model_dir=args.lstm_model_dir,
        rl_model_path=args.rl_model_path,
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
    )

    # — rutas que KFP recogerá —
    Path("/tmp/backtest_dir.txt").write_text(out_dir)
    Path("/tmp/kfp_metrics.json").write_text(Path(kfp_metrics_file).read_text())
