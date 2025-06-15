# src/components/backtest/task.py
"""
Back-test comparativo: genera reporte QuantStats, CSV de trades,
JSON de métricas y artefacto KFP para Vertex AI.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import quantstats as qs
import tensorflow as tf
from tensorflow.keras import models

from src.shared import constants, gcs_utils, indicators

# ───────────────────────── Configuración global ────────────────────────────
os.environ["QS_NO_IPYTHON"] = "1"  # evita dependencia de IPython en QuantStats
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
qs.extend_pandas()

# ─────────────────────────────── Helpers ───────────────────────────────────
def _load_artifacts(
    lstm_model_dir: str, filter_model_path: str
) -> Tuple[tf.keras.Model, object, dict, lgb.LGBMClassifier, dict]:
    """Carga modelo LSTM, scaler, hp y filtro LightGBM desde GCS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model = tf.keras.models.load_model(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.keras", tmp),
            compile=False,
        )
        scaler = joblib.load(gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp))
        hp = json.loads(gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp).read_text())
        filter_model = joblib.load(gcs_utils.download_gcs_file(f"{filter_model_path}/filter_model.pkl", tmp))
        filter_params = json.loads(
            gcs_utils.download_gcs_file(f"{filter_model_path}/filter_params.json", tmp).read_text()
        )
    logger.info("✔ Artefactos LSTM y filtro LightGBM cargados.")
    return model, scaler, hp, filter_model, filter_params


def _prepare_backtest_data(features_path: str, hp: dict) -> pd.DataFrame:
    """Carga hold-out, recalcula indicadores y asegura datos limpios."""
    local = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    df_raw = pd.read_parquet(local)
    if df_raw.empty:
        raise ValueError("Hold-out Parquet vacío.")
    df_ind = indicators.build_indicators(df_raw, hp, atr_len=14, drop_na=True)
    if df_ind.isna().any().any():
        raise ValueError("Persisten NaNs tras calcular indicadores.")
    logger.info("✔ Datos de back-test: %s filas", len(df_ind))
    return df_ind.reset_index(drop=True)


def _generate_features_for_filter(
    df: pd.DataFrame,
    lstm_model: tf.keras.Model,
    scaler,
    hp: dict,
    pair: str,
):
    """Genera embeddings LSTM + features adicionales para el filtro."""
    feature_cols = list(scaler.feature_names_in_)
    X_scaled = scaler.transform(df[feature_cols].values)
    X_seq = np.stack([X_scaled[i - hp["win"] : i] for i in range(hp["win"], len(X_scaled))]).astype(
        np.float32
    )

    emb_model = models.Model(lstm_model.input, lstm_model.layers[-2].output)
    preds = lstm_model.predict(X_seq, verbose=0, batch_size=1024)
    embs = emb_model.predict(X_seq, verbose=0, batch_size=1024)

    df_aligned = df.iloc[hp["win"] :].copy().reset_index()
    features_df = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])])
    features_df["pred_up"] = preds[:, 0]
    features_df["pred_down"] = preds[:, 1]
    features_df["pred_diff"] = preds[:, 0] - preds[:, 1]

    tick = 0.01 if pair.endswith("JPY") else 0.0001
    atr = df_aligned["atr_14"].values
    features_df["sl_pips"] = np.where(
        features_df["pred_up"] > features_df["pred_down"],
        hp["min_thr_up"] * atr / tick,
        hp["min_thr_dn"] * atr / tick,
    )
    features_df["tp_pips"] = features_df["sl_pips"] * hp["rr"]
    features_df["rr_ratio"] = hp["rr"]

    df_aligned["timestamp"] = pd.to_datetime(df_aligned["timestamp"], unit="ms")
    features_df["hour"] = df_aligned["timestamp"].dt.hour
    features_df["day_of_week"] = df_aligned["timestamp"].dt.dayofweek

    return features_df, preds[:, 0], preds[:, 1]


def _run_backtest_simulation_from_signals(
    df_aligned: pd.DataFrame,
    up_preds: np.ndarray,
    dn_preds: np.ndarray,
    hp: dict,
    pair: str,
    accept_mask: np.ndarray,
) -> pd.DataFrame:
    """Simula operaciones según señales aceptadas."""
    trades = []
    dq = deque(maxlen=hp["smooth_win"])
    tick = 0.01 if pair.endswith("JPY") else 0.0001
    horizon = hp["horizon"]
    closes = df_aligned.close.values

    for i in range(len(up_preds) - horizon):
        u, d = up_preds[i], dn_preds[i]
        raw_dir = 1 if u > d else -1
        diff = abs(u - d)

        cond_base = ((raw_dir == 1 and u >= hp["min_thr_up"]) or (raw_dir == -1 and d >= hp["min_thr_dn"])) and (
            diff >= hp["delta_min"]
        )

        dq.append(raw_dir if cond_base else 0)
        buys, sells = dq.count(1), dq.count(-1)
        base_signal = 1 if buys > hp["smooth_win"] // 2 else -1 if sells > hp["smooth_win"] // 2 else 0
        final_signal = base_signal if accept_mask[i] else 0

        if final_signal != 0:
            entry_time = pd.to_datetime(df_aligned["timestamp"].iloc[i], unit="ms")
            exit_time = pd.to_datetime(df_aligned["timestamp"].iloc[i + horizon], unit="ms")
            pnl_pips = ((closes[i + horizon] - closes[i]) / tick) * final_signal - constants.SPREADS_PIP.get(pair, 0.8)
            trades.append({"entry_time": entry_time, "exit_time": exit_time, "direction": final_signal, "pnl_pips": pnl_pips})

    return pd.DataFrame(trades)


def _calculate_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula métricas QS a partir del DataFrame de trades."""
    if trades_df.empty:
        return {k: 0.0 for k in ["trades", "win_rate", "total_pips", "sharpe", "calmar", "max_drawdown"]}

    idx = pd.to_datetime(trades_df["exit_time"], utc=False)
    returns = pd.Series(trades_df["pnl_pips"] / 10000, index=idx)

    metrics = {
        "trades": len(returns),
        "win_rate": float((returns > 0).mean()),
        "total_pips": float(trades_df["pnl_pips"].sum()),
        "avg_pips": float(trades_df["pnl_pips"].mean()),
        "sharpe": float(qs.stats.sharpe(returns, periods=252 * (24 * 60 / 15))),
        "calmar": float(qs.stats.calmar(returns)),
        "max_drawdown": float(qs.stats.max_drawdown(returns)),
    }
    return {k: (v if np.isfinite(v) else 0.0) for k, v in metrics.items()}


# ─────────────────────────── Orquestador principal ─────────────────────────
def run_backtest(
    *,
    lstm_model_dir: str,
    filter_model_path: str,
    features_path: str,
    pair: str,
    timeframe: str,
) -> Tuple[str, str]:
    """Ejecuta back-test, sube artefactos y devuelve ruta + metrics path."""
    lstm_model, scaler, hp, filter_model, filter_params = _load_artifacts(lstm_model_dir, filter_model_path)
    df_bt = _prepare_backtest_data(features_path, hp)
    df_aligned = df_bt.iloc[hp["win"] :].copy().reset_index(drop=True)
    # ➜ Garantizamos que el timestamp sea datetime para la simulación
    df_aligned["timestamp"] = pd.to_datetime(df_aligned["timestamp"], unit="ms")

    features_df, up_preds, dn_preds = _generate_features_for_filter(df_bt, lstm_model, scaler, hp, pair)

    accept_mask = filter_model.predict_proba(features_df)[:, 1] >= filter_params["best_threshold"]

    logger.info("Simulando estrategia BASE …")
    base_trades = _run_backtest_simulation_from_signals(
        df_aligned, up_preds, dn_preds, hp, pair, np.ones_like(accept_mask, dtype=bool)
    )
    logger.info("Simulando estrategia FILTRADA …")
    filtered_trades = _run_backtest_simulation_from_signals(df_aligned, up_preds, dn_preds, hp, pair, accept_mask)

    metrics = {"base": _calculate_metrics(base_trades), "filtered": _calculate_metrics(filtered_trades)}
    logger.info("Métricas filtrada: %s", metrics["filtered"])

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir = f"{constants.BACKTEST_RESULTS_PATH}/{pair}/{timeframe}/{ts}"

    # -------------------------- manejo de archivos --------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if not base_trades.empty:
            base_trades.to_csv(tmp / "trades_base.csv", index=False)
            gcs_utils.upload_gcs_file(tmp / "trades_base.csv", f"{output_dir}/trades_base.csv")

        if not filtered_trades.empty:
            df_rep = filtered_trades.copy()
            df_rep["returns"] = df_rep["pnl_pips"] / 10000
            df_rep["entry_time"] = pd.to_datetime(df_rep["entry_time"])  # índice datetime para QuantStats
            df_rep.set_index("entry_time", inplace=True)

            report_path = tmp / "report_filtered.html"
            qs.reports.html(df_rep["returns"], output=str(report_path), title=f"{pair} Filtered Strategy")
            gcs_utils.upload_gcs_file(report_path, f"{output_dir}/report_filtered.html")

            df_rep.to_csv(tmp / "trades_filtered.csv")
            gcs_utils.upload_gcs_file(tmp / "trades_filtered.csv", f"{output_dir}/trades_filtered.csv")

        (tmp / "metrics.json").write_text(json.dumps(metrics, indent=4))
        gcs_utils.upload_gcs_file(tmp / "metrics.json", f"{output_dir}/metrics.json")

        kfp_dict = {"metrics": [{"name": f"filtered-{k}", "numberValue": v} for k, v in metrics["filtered"].items()]}
        kfp_tmp = tmp / "kfp_metrics.json"
        kfp_tmp.write_text(json.dumps(kfp_dict))

        Path("/tmp/backtest_dir.txt").write_text(output_dir)
        Path("/tmp/kfp_metrics.json").write_text(kfp_tmp.read_text())

    logger.info("🎯 Back-test finalizado. Resultados en %s", output_dir)
    return output_dir, "/tmp/kfp_metrics.json"


# ─────────────────────────────── CLI ───────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Back-test Task")
    p.add_argument("--lstm-model-dir", required=True)
    p.add_argument("--filter-model-path", required=True)
    p.add_argument("--features-path", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    args = p.parse_args()

    out_dir, kfp_metrics_path = run_backtest(
        lstm_model_dir=args.lstm_model_dir,
        filter_model_path=args.filter_model_path,
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
    )

    logger.info("Back-test completado. Dir: %s", out_dir)
    logger.info("KFP metrics en: %s", kfp_metrics_path)
