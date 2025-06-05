#!/usr/bin/env python3
"""
Genera el dataset (.npz) que usará el agente PPO para filtrar
las señales del LSTM.

Contiene:
    obs   : [n, 2 + emb_dim]   → (up_pred, dn_pred, embedding…)
    raw   : [n,]               → señal discreta 1 / -1 / 0
    closes: [n,]               → precios de cierre
"""

# ───────────────── imports estándar ──────────────────────
import os, sys, json, warnings, random, tempfile, logging
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd, joblib
import tensorflow as tf
from tensorflow.keras import models, mixed_precision
from collections import deque
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.preprocessing import RobustScaler

from indicators import build_indicators

# ───────────────── configuración global ───────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

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

# ───────────────── helpers GCS ────────────────────────────
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
pa.add_argument("--model",    required=True, help="gs://… modelo LSTM (.h5)")
pa.add_argument("--scaler",   required=True, help="gs://… scaler (.pkl)")
pa.add_argument("--features", required=True, help="gs://… datos OHLC (.parquet)")
pa.add_argument("--params",   required=True, help="gs://… best_params.json")
pa.add_argument("--output",   required=True, help="gs://…/ppo_input_data.npz")
args = pa.parse_args()

# ───────────────── carga de artefactos ────────────────────
lstm_model = tf.keras.models.load_model(maybe_local(args.model), compile=False)
emb_model  = models.Model(lstm_model.input, lstm_model.layers[-2].output)

scaler     = joblib.load(maybe_local(args.scaler))
hp         = json.loads(maybe_local(args.params).read_text())
PAIR       = hp["pair"]
tick       = 0.01 if PAIR.endswith("JPY") else 0.0001
ATR_LEN    = 14

df_raw = pd.read_parquet(maybe_local(args.features)).reset_index(drop=True)
if "timestamp" in df_raw.columns:
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")

# ───────────────── indicadores & limpieza ─────────────────
df = build_indicators(df_raw.copy(), hp, ATR_LEN)
atr_col = f"atr_{ATR_LEN}"
if atr_col not in df or df[atr_col].isna().all():
    logger.critical("ATR inexistente o todo NaN. Abortando.")
    sys.exit(1)

df[atr_col] = df[atr_col].fillna(method="bfill").fillna(method="ffill")
if df[atr_col].isna().any():
    logger.critical("ATR todavía contiene NaNs tras rellenar. Abortando.")
    sys.exit(1)

# ───────────────── escalado & secuencias ──────────────────
features_for_scaler = (
    [c for c in scaler.feature_names_in_ if c in df.columns] 
    if hasattr(scaler, "feature_names_in_") else
    [c for c in df.columns if c not in ("timestamp", atr_col) and pd.api.types.is_numeric_dtype(df[c])]
)

X_raw = df[features_for_scaler].select_dtypes(include=np.number)
if X_raw.empty:
    logger.critical("No hay columnas numéricas para escalar. Abortando.")
    sys.exit(1)

X_scaled = scaler.transform(X_raw)

def make_seq(arr, win):
    if len(arr) <= win:
        return np.empty((0,))
    return np.stack([arr[i - win : i] for i in range(win, len(arr))]).astype(np.float32)

win = hp["win"]
X_seq = make_seq(X_scaled, win)
if X_seq.shape[0] == 0:
    logger.critical("No se pudieron generar secuencias suficientes. Abortando.")
    sys.exit(1)

closes   = df.close.values[win:]
atr_pips = df[atr_col].values[win:] / tick

# ───────────────── predicción + embedding ─────────────────
with tf.device("/GPU:0" if gpus else "/CPU:0"):
    preds = lstm_model.predict(X_seq, verbose=0).astype(np.float32)
    embs  = emb_model.predict(X_seq, verbose=0).astype(np.float32)

pred_up, pred_dn = preds[:, 0], preds[:, 1]
OBS = np.hstack([preds, embs]).astype(np.float32)

# ───────────────── generación de raw_signal ───────────────
raw_signal = np.zeros(len(closes), dtype=np.int8)
dq = deque(maxlen=hp["smooth_win"])

for i, (u, d, atr) in enumerate(zip(pred_up, pred_dn, atr_pips)):
    mag, diff = max(u, d), abs(u - d)
    raw_dir   = 1 if u > d else -1
    cond = (
        ((raw_dir == 1 and mag >= hp["min_thr_up"]) or
         (raw_dir == -1 and mag >= hp["min_thr_dn"]))
        and diff >= hp["delta_min"]
    )
    dq.append(raw_dir if cond else 0)
    buys, sells = dq.count(1), dq.count(-1)
    raw_signal[i] = 1 if buys > hp["smooth_win"] // 2 else -1 if sells > hp["smooth_win"] // 2 else 0

# ───────────────── guardado en npz ────────────────────────
with tempfile.TemporaryDirectory() as td:
    npz_local = Path(td) / "ppo_input_data.npz"
    np.savez(npz_local, obs=OBS, raw=raw_signal.astype(np.int8), closes=closes.astype(np.float32))
    upload_gs(npz_local, args.output)

logger.info("🎉 Archivo NPZ subido a %s", args.output)
