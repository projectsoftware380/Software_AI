# src/components/hyperparam_tuning/task.py
"""
Optuna HPO para el modelo LSTM.

Flujo:
1. Calcula indicadores y secuencias.
2. Entrena un LSTM por trial y hace un back-test rápido.
3. Guarda best_params.json en GCS y expone métricas para KFP.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
import tempfile
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import gcsfs
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, optimizers

from src.shared import constants, gcs_utils, indicators

# ───────────────────────────── logging ──────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ───────── reproducibilidad + (opcional) configuración GPU ──────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("🚀 GPU(s) detectadas – memoria dinámica y mixed-precision activadas.")
    else:
        raise RuntimeError("No GPU found")
except Exception as exc:
    logger.warning("⚠️  No se utilizará GPU (%s). Continuando con CPU; el HPO será más lento.",
                   exc if isinstance(exc, RuntimeError) else "error al configurar GPU")
    gpus = []

# ────────────────────── housekeeping GCS (opt) ──────────────────────
def _keep_only_latest_version(base_gcs_prefix: str) -> None:
    try:
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
        base_gcs_prefix = base_gcs_prefix.rstrip("/") + "/"
        timestamp_re = re.compile(r"/(\d{14})/?$")
        dirs = [d for d in fs.ls(base_gcs_prefix) if fs.isdir(d) and timestamp_re.search(d)]
        if len(dirs) <= 1:
            return
        dirs.sort(key=lambda p: timestamp_re.search(p).group(1), reverse=True)
        for old_dir in dirs[1:]:
            logger.info("🗑️  Borrando versión anterior: gs://%s", old_dir)
            fs.rm(old_dir, recursive=True)
    except Exception as exc:
        logger.warning("No se pudo limpiar versiones antiguas: %s", exc)

# ───────────────────────── helpers de modelo  ───────────────────────
def make_model(inp_shape, lr, dr, filt, units, heads):
    x = inp = layers.Input(shape=inp_shape, dtype=tf.float32)
    x = layers.Conv1D(filt, 3, padding="same", activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizers.Adam(lr), loss="mae")
    return model

def quick_bt(pred, closes, atr, rr, up_thr, dn_thr, delta_min, smooth_win, tick):
    net, pos, dq = 0.0, False, deque(maxlen=smooth_win)
    entry_price = direction = sl = tp = 0.0
    for (u, d), price, atr_i in zip(pred, closes, atr):
        mag, diff = max(u, d), abs(u - d)
        raw = 1 if u > d else -1
        cond = ((raw == 1 and mag >= up_thr) or (raw == -1 and mag >= dn_thr)) and diff >= delta_min
        dq.append(raw if cond else 0)
        buys, sells = dq.count(1), dq.count(-1)
        signal = 1 if buys > smooth_win // 2 else -1 if sells > smooth_win // 2 else 0

        if not pos and signal:
            pos, entry_price, direction = True, price, signal
            sl = (up_thr if direction == 1 else dn_thr) * atr_i
            tp = rr * sl
            continue

        if pos:
            pnl = ((price - entry_price) if direction == 1 else (entry_price - price)) / tick
            if pnl >= tp or pnl <= -sl:
                net += tp if pnl >= tp else -sl
                pos = False
    return net

# ────────────────────────── función principal ───────────────────────
def run_optimization(*,
                     features_path: str,
                     pair: str,
                     timeframe: str,
                     n_trials: int,
                     output_gcs_path: str,
                     metrics_output_file: Path,
                     cleanup_old_versions: bool = True) -> None:

    logger.info("🚀 Iniciando HPO (%s / %s) con %d trials…", pair, timeframe, n_trials)

    local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    df_raw = pd.read_parquet(local_features_path)

    if "timestamp" not in df_raw.columns:
        raise ValueError("La columna 'timestamp' es obligatoria y no se encontró.")
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")

    def objective(trial: optuna.Trial) -> float:
        tf.keras.backend.clear_session()
        gc.collect()

        tick = 0.01 if pair.endswith("JPY") else 0.0001
        atr_len = 14

        p = {
            "horizon":    trial.suggest_int("horizon", 10, 30),
            "rr":         trial.suggest_float("rr", 1.5, 3.0),
            "min_thr_up": trial.suggest_float("min_thr_up", 0.5, 2.0),
            "min_thr_dn": trial.suggest_float("min_thr_dn", 0.5, 2.0),
            "delta_min":  trial.suggest_float("delta_min", 0.01, 0.5),
            "smooth_win": trial.suggest_int("smooth_win", 1, 5),
            "win":        trial.suggest_int("win", 20, 60),
            "lr":         trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "dr":         trial.suggest_float("dr", 0.1, 0.5),
            "filt":       trial.suggest_categorical("filt",  [16, 32, 64]),
            "units":      trial.suggest_categorical("units", [32, 64, 128]),
            "heads":      trial.suggest_categorical("heads", [2, 4, 8]),
            "sma_len":    trial.suggest_categorical("sma_len", [20, 40, 60]),
            "rsi_len":    trial.suggest_categorical("rsi_len", [7, 14, 21]),
            "macd_fast":  trial.suggest_categorical("macd_fast", [8, 12]),
            "macd_slow":  trial.suggest_categorical("macd_slow", [21, 26]),
            "stoch_len":  trial.suggest_categorical("stoch_len", [14, 21]),
        }

        df_ind = indicators.build_indicators(df_raw.copy(), p, atr_len=atr_len)

        atr = df_ind[f"atr_{atr_len}"].values / tick
        close = df_ind.close.values
        fut_close = np.roll(close, -p["horizon"])
        fut_close[-p["horizon"]:] = np.nan
        diff = (fut_close - close) / tick
        up = np.maximum(diff, 0) / atr
        dn = np.maximum(-diff, 0) / atr

        mask = (~np.isnan(diff)) & (~np.isnan(atr))
        if mask.sum() < 1_000:
            return -1e8

        feature_cols = [c for c in df_ind.columns
                        if c != "timestamp" and not c.startswith("atr_")]
        X_raw = df_ind.loc[mask, feature_cols].select_dtypes(include=np.number)
        if X_raw.empty or X_raw.shape[0] <= p["win"]:
            return -1e8

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_raw)

        X_seq = np.stack(
            [X_scaled[i - p["win"]: i] for i in range(p["win"], len(X_scaled))]
        ).astype(np.float32)

        if X_seq.shape[0] < 500:
            return -1e8

        y_up_seq = up[mask][p["win"]:]
        y_dn_seq = dn[mask][p["win"]:]
        closes_seq = close[mask][p["win"]:]
        atr_seq = atr[mask][p["win"]:]

        # ===== CORRECCIÓN =====
        # Se añaden las variables que faltaban (closes_tr, atr_tr)
        # para desempaquetar correctamente los 10 arrays que devuelve la función.
        X_tr, X_val, y_up_tr, y_up_val, y_dn_tr, y_dn_val, closes_tr, closes_val, atr_tr, atr_val = train_test_split(
            X_seq, y_up_seq, y_dn_seq, closes_seq, atr_seq,
            test_size=0.2, shuffle=False
        )

        model = make_model(
            X_tr.shape[1:], p["lr"], p["dr"],
            p["filt"], p["units"], p["heads"]
        )
        model.fit(
            X_tr,
            np.column_stack([y_up_tr, y_dn_tr]),
            validation_data=(X_val, np.column_stack([y_up_val, y_dn_val])),
            epochs=15,
            batch_size=64,
            verbose=0,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

        score = quick_bt(
            model.predict(X_val, verbose=0),
            closes_val, atr_val,
            p["rr"], p["min_thr_up"], p["min_thr_dn"],
            p["delta_min"], p["smooth_win"], tick
        )
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {
        **study.best_params,
        "pair": pair,
        "timeframe": timeframe,
        "features_path": features_path,
        "optimization_timestamp_utc": datetime.utcnow().isoformat(),
        "best_trial_score": study.best_value,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_json = Path(tmpdir) / "best_params.json"
        tmp_json.write_text(json.dumps(best_params, indent=2))
        gcs_utils.upload_gcs_file(tmp_json, output_gcs_path)

    metrics_output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_file.write_text(json.dumps({
        "metrics": [{
            "name": "optuna-best-trial-score",
            "numberValue": best_params["best_trial_score"] or -1e9,
            "format": "RAW"
        }]
    }))
    logger.info("✅ HPO completado. Best score: %.2f", study.best_value)

    if cleanup_old_versions:
        _keep_only_latest_version(f"{constants.LSTM_PARAMS_PATH}/{pair}/{timeframe}/")

# ────────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    cli = argparse.ArgumentParser("Optuna HPO task")
    cli.add_argument("--features-path", required=True)
    cli.add_argument("--pair", required=True)
    cli.add_argument("--timeframe", required=True)
    cli.add_argument("--n-trials", type=int, default=constants.DEFAULT_N_TRIALS)
    cli.add_argument("--cleanup-old-versions",
                     type=lambda x: str(x).lower() == "true", default=True)
    cli.add_argument("--best-params-path-output", type=Path, required=True)
    cli.add_argument("--optimization-metrics-output", type=Path, required=True)
    args = cli.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_gcs = (f"{constants.LSTM_PARAMS_PATH}/{args.pair}/{args.timeframe}/"
               f"{ts}/best_params.json")

    run_optimization(
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
        n_trials=args.n_trials,
        output_gcs_path=out_gcs,
        metrics_output_file=args.optimization_metrics_output,
        cleanup_old_versions=args.cleanup_old_versions,
    )

    args.best_params_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.best_params_path_output.write_text(out_gcs)