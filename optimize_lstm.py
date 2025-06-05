#!/usr/bin/env python3
"""
Busca los mejores hiperparámetros para un modelo LSTM usando Optuna
y guarda un `best_params.json` en GCS o local.

Cambios destacados
------------------
* Valida que **ATR** exista y quede sin NaNs antes de entrenar.
* Comprueba DataFrames vacíos, desajustes de longitud y secuencias insuficientes.
* Registro exhaustivo (logging) para depuración en Vertex AI.
* La ruta `--output` apunta directamente al archivo JSON final.
"""

# ───────────────── imports estándar ──────────────────────
import os, sys, json, random, warnings, tempfile, gc
from pathlib import Path
import argparse
from datetime import datetime
from collections import deque
import logging

# ───────────────── configuración de logging ───────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ───────────────── libs científicas ───────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
if not hasattr(np, "NaN"):      # compat. NumPy < 1.24
    np.NaN = np.nan

import pandas as pd, joblib, optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision

from google.cloud import storage
from google.oauth2 import service_account

from indicators import build_indicators   # módulo propio

# ───────────────── reproducibilidad & GPU ─────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
mixed_precision.set_global_policy("mixed_float16")

# ───────────────── helpers de GCS ─────────────────────────
def _gcs_client():
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        return storage.Client(credentials=creds)
    return storage.Client()

def download_gs(uri: str) -> Path:
    bucket, blob = uri[5:].split("/", 1)
    local = Path(tempfile.mkdtemp()) / Path(blob).name
    _gcs_client().bucket(bucket).blob(blob).download_to_filename(local)
    return local

def upload_gs(local: Path, uri: str):
    bucket, blob = uri[5:].split("/", 1)
    _gcs_client().bucket(bucket).blob(blob).upload_from_filename(str(local))

def maybe_local(path: str) -> Path:
    return download_gs(path) if path.startswith("gs://") else Path(path)

# ───────────────── CLI ────────────────────────────────────
pa = argparse.ArgumentParser()
pa.add_argument("--features",   required=True, help="gs://… parquet con OHLC+indicadores básicos")
pa.add_argument("--pair",       required=True, help="Ej.: EURUSD")
pa.add_argument("--timeframe",  required=True, help="Ej.: 15minute")
pa.add_argument("--output",     required=True, help="gs://…/best_params.json")
pa.add_argument("--n-trials",   type=int, default=25)
args = pa.parse_args()

PAIR, TF = args.pair, args.timeframe
tick = 0.01 if PAIR.endswith("JPY") else 0.0001
ATR_LEN = 14
EPOCHS_OPT, BATCH_OPT = 15, 64

# ───────────────── carga base de datos ────────────────────
df_raw = pd.read_parquet(maybe_local(args.features)).reset_index(drop=True)
if "timestamp" in df_raw.columns:
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")

# ───────────────── modelo LSTM base ───────────────────────
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

def quick_bt(pred, closes, atr, rr, up_thr, dn_thr, delta_min, smooth_win):
    net, pos, dq = 0.0, False, deque(maxlen=smooth_win)
    for (u, d), price, atr_i in zip(pred, closes, atr):
        mag, diff = max(u, d), abs(u - d)
        raw = 1 if u > d else -1
        cond = ((raw == 1 and mag >= up_thr) or (raw == -1 and mag >= dn_thr)) and diff >= delta_min
        dq.append(raw if cond else 0)
        buys, sells = dq.count(1), dq.count(-1)
        signal = 1 if buys > smooth_win // 2 else -1 if sells > smooth_win // 2 else 0
        if not pos and signal:
            pos, entry, sd = True, price, signal
            sl = (up_thr if sd == 1 else dn_thr) * atr_i
            tp = rr * sl
            continue
        if pos and signal:
            sl = min(sl, (up_thr if signal == 1 else dn_thr) * atr_i)
            tp = max(tp, rr * sl)
        if pos:
            pnl = (price - entry) / tick if sd == 1 else (entry - price) / tick
            if pnl >= tp or pnl <= -sl:
                net += tp if pnl >= tp else -sl
                pos = False
    return net

# ───────────────── función objective ───────────────────────
def objective(trial: optuna.trial.Trial) -> float:
    p = {
        # hiperparámetros trading
        "horizon":    trial.suggest_int("horizon", 10, 30),
        "rr":         trial.suggest_float("rr", 1.5, 3.0),
        "min_thr_up": trial.suggest_float("min_thr_up", 0.5, 2.0),
        "min_thr_dn": trial.suggest_float("min_thr_dn", 0.5, 2.0),
        "delta_min":  trial.suggest_float("delta_min", 0.01, 0.5),
        "smooth_win": trial.suggest_int("smooth_win", 1, 5),
        # hiperparámetros LSTM
        "win":        trial.suggest_int("win", 20, 60),
        "lr":         trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "dr":         trial.suggest_float("dr", 0.1, 0.5),
        "filt":       trial.suggest_categorical("filt", [16, 32, 64]),
        "units":      trial.suggest_categorical("units", [32, 64, 128]),
        "heads":      trial.suggest_categorical("heads", [2, 4, 8]),
        # hiperparámetros indicadores
        "sma_len":    trial.suggest_categorical("sma_len",  [20, 40, 60]),
        "rsi_len":    trial.suggest_categorical("rsi_len",  [7, 14, 21]),
        "macd_fast":  trial.suggest_categorical("macd_fast", [8, 12]),
        "macd_slow":  trial.suggest_categorical("macd_slow", [21, 26]),
        "stoch_len":  trial.suggest_categorical("stoch_len", [14, 21]),
    }

    # ---------- indicadores ----------
    df = build_indicators(df_raw.copy(), p, ATR_LEN)
    atr_col = f"atr_{ATR_LEN}"
    if atr_col not in df or df[atr_col].isna().all():
        logger.warning("ATR ausente o todo NaN → trial penalizado")
        return -1e9
    df[atr_col] = df[atr_col].fillna(method="bfill").fillna(method="ffill")
    if df[atr_col].isna().any():
        logger.warning("ATR todavía contiene NaNs → trial penalizado")
        return -1e9

    # ---------- etiquetas ----------
    atr = df[atr_col].values / tick
    cls = df.close.values
    fut = np.roll(cls, -p["horizon"])
    fut[-p["horizon"]:] = np.nan
    diff = (fut - cls) / tick
    up = np.maximum(diff, 0) / atr
    dn = np.maximum(-diff, 0) / atr
    mask = (~np.isnan(diff)) & (~np.isnan(atr))

    if mask.sum() < 1000:
        return -1e8           # muy pocos datos útiles

    feats = [c for c in df.columns if c not in (atr_col, "timestamp")]
    X_raw = df.loc[mask, feats].select_dtypes(include=np.number)
    if X_raw.empty or X_raw.shape[0] <= p["win"]:
        return -1e8

    y_up, y_dn = up[mask], dn[mask]
    cls_m, atr_m = cls[mask], atr[mask]

    # ---------- escalado + secuencias ----------
    sc = RobustScaler()
    X_scaled = sc.fit_transform(X_raw)

    def seq(arr, w):
        if len(arr) < w:
            return np.empty((0,))
        return np.stack([arr[i - w : i] for i in range(w, len(arr))]).astype(np.float32)

    X_seq = seq(X_scaled, p["win"])
    if X_seq.shape[0] < 500:
        return -1e8

    y_up = y_up[p["win"]:]
    y_dn = y_dn[p["win"]:]
    cls_m = cls_m[p["win"]:]
    atr_m = atr_m[p["win"]:]

    X_tr, X_val, up_tr, up_val, dn_tr, dn_val, cls_tr, cls_val, atr_tr, atr_val = train_test_split(
        X_seq, y_up, y_dn, cls_m, atr_m, test_size=0.2, shuffle=False
    )

    model = make_model(X_tr.shape[1:], p["lr"], p["dr"], p["filt"], p["units"], p["heads"])
    model.fit(
        X_tr, np.vstack([up_tr, dn_tr]).T,
        validation_data=(X_val, np.vstack([up_val, dn_val]).T),
        epochs=EPOCHS_OPT,
        batch_size=BATCH_OPT,
        verbose=0,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    score = quick_bt(
        model.predict(X_val, verbose=0),
        cls_val,
        atr_val,
        p["rr"], p["min_thr_up"], p["min_thr_dn"],
        p["delta_min"], p["smooth_win"]
    )

    tf.keras.backend.clear_session()
    gc.collect()
    return score

# ───────────────── ejecución de Optuna ──────────────────────
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

best = study.best_params
best.update({
    "pair": PAIR,
    "timeframe": TF,
    "features_path": args.features,
    "timestamp": datetime.utcnow().isoformat(),
})

best_json = json.dumps(best, indent=2)
with tempfile.TemporaryDirectory() as td:
    tmpfile = Path(td) / "best_params.json"
    tmpfile.write_text(best_json)
    upload_gs(tmpfile, args.output)

logger.info("✅ best_params.json guardado en %s", args.output)
