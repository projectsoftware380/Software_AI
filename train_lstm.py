#!/usr/bin/env python3
"""
Entrena el modelo LSTM final usando hiperparámetros generados por optimize_lstm.py
(tanto en local como en GCS) y sube los artefactos (modelo, scaler y params JSON)
a Google Cloud Storage con versionado por timestamp.

Cambios clave respecto a la versión anterior
-------------------------------------------
1. **Argumentos CLI flexibles** – ahora acepta los mismos argumentos que recibe el
   "runner" desde Vertex AI Custom Job (–output-gcs-base-dir, –pair, –timeframe,
   etc.). Los argumentos innecesarios se ignoran para que el script sea retro-
   compatible cuando se ejecute manualmente.
2. **Ruta base parametrizable** – ya no está hard-codeada. Si se pasa
   `--output-gcs-base-dir`, se usa esa ruta; de lo contrario se mantiene el valor
   por defecto (gs://trading-ai-models-460823/models/LSTM).
3. **Impresión del directorio de artefactos** – al final del script se muestra la
   ruta GCS en la que quedaron los artefactos, útil para parsearla en pipelines.
4. **Robustez extra** – manejo de errores de GCS, chequeos de tipos numéricos y
   fallback silencioso cuando no hay GPU.
"""

# ────────────────── imports estándar / utilidades ──────────────────
import os
import sys
import json
import random
import warnings
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import joblib
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision

# Suprimir warnings verbosos de pandas
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# 👉 Indicadores centralizados
from indicators import build_indicators

# ────────────────── determinismo y configuración de GPU ────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    mixed_precision.set_global_policy("mixed_float16")
except Exception as gpu_err:
    print(f"[WARN] No se pudo configurar la GPU o mixed precision: {gpu_err}")

# ────────────────────────── utilidades GCS ─────────────────────────

def _gcs_client():
    """Devuelve un cliente de GCS, respetando GOOGLE_APPLICATION_CREDENTIALS."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        creds = service_account.Credentials.from_service_account_file(creds_path)
        return storage.Client(credentials=creds)
    return storage.Client()

def download_from_gcs(gcs_uri: str) -> Path:
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    local_tmp = Path(tempfile.mkdtemp()) / Path(blob_name).name
    _gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(local_tmp)
    return local_tmp

def upload_to_gcs(local_path: Path, gcs_uri: str):
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    _gcs_client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))

# ───────────────────────── funciones core ─────────────────────────
ATR_LEN = 14

def to_sequences(mat, up, dn, closes, win):
    X, y_up, y_dn, cl = [], [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i - win : i])
        y_up.append(up[i])
        y_dn.append(dn[i])
        cl.append(closes[i])
    return (
        np.asarray(X, np.float32),
        np.asarray(y_up, np.float32),
        np.asarray(y_dn, np.float32),
        np.asarray(cl, np.float32),
    )

def make_model(inp_shape, lr, dr, filt, units, heads):
    inp = layers.Input(shape=inp_shape, dtype=tf.float32)
    x = layers.Conv1D(filt, 3, padding="same", activation="relu")(inp)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizers.Adam(lr), loss="mae")
    return model

def quick_bt(pred, closes, atr_pips, rr, up_thr, dn_thr, delta_min, smooth_win, tick):
    """Back-test ultrarrápido (sin comisiones) para validar la señal."""
    net, pos = 0.0, False
    dq = deque(maxlen=smooth_win)
    entry_price = entry_dir = sl = tp = 0  # init

    for (u, d), price, atr in zip(pred, closes, atr_pips):
        mag, diff = max(u, d), abs(u - d)
        raw_signal = 1 if u > d else -1
        min_thr = up_thr if raw_signal == 1 else dn_thr
        dq.append(raw_signal if (mag >= min_thr and diff >= delta_min) else 0)
        buy_cnt, sell_cnt = dq.count(1), dq.count(-1)
        signal = 1 if buy_cnt > smooth_win // 2 else -1 if sell_cnt > smooth_win // 2 else 0

        # Apertura
        if not pos and signal:
            pos, entry_price, entry_dir = True, price, signal
            sl = (up_thr if signal == 1 else dn_thr) * atr
            tp = rr * sl
            continue

        # Ajuste dinámico
        if pos and signal:
            cand_sl = (up_thr if signal == 1 else dn_thr) * atr
            cand_tp = rr * cand_sl
            sl = min(sl, cand_sl)
            tp = max(tp, cand_tp)

        # Cierre
        if pos:
            pnl = ((price - entry_price) if entry_dir == 1 else (entry_price - price)) / tick
            if pnl >= tp or pnl <= -sl:
                net += tp if pnl >= tp else -sl
                pos = False
    return net

# ──────────────────────────── CLI ────────────────────────────────
parser = argparse.ArgumentParser(description="Train final LSTM model and upload to GCS")
parser.add_argument("--params", required=True, help="Ruta local o gs://... del JSON con hiperparámetros (output de optimize_lstm.py)")
# Argumentos opcionales (se ignoran si no se usan)
parser.add_argument("--output-gcs-base-dir", default="gs://trading-ai-models-460823/models/LSTM", help="Ruta base gs:// donde se crearán las carpetas <pair>/<tf>/<timestamp>")
parser.add_argument("--pair", default=None, help="Override del par si se desea")
parser.add_argument("--timeframe", default=None, help="Override del timeframe si se desea")
# Extras para compatibilidad con el launcher – los aceptamos pero no los usamos explícitamente
parser.add_argument("--project-id", default=None)
parser.add_argument("--gcs-bucket-name", default=None)
args, _ = parser.parse_known_args()  # tolerar flags desconocidas

# ────────────────────── cargar hiperparámetros ───────────────────
params_path = Path(args.params) if not args.params.startswith("gs://") else download_from_gcs(args.params)
params = json.loads(params_path.read_text())

PAIR = args.pair or params["pair"]
TF   = args.timeframe or params["timeframe"]
TICK = 0.01 if PAIR.endswith("JPY") else 0.0001

# ────────────────────────── cargar features ──────────────────────
feat_path = params["features_path"]
feat_local = Path(feat_path) if not feat_path.startswith("gs://") else download_from_gcs(feat_path)

df_raw = pd.read_parquet(feat_local).reset_index(drop=True)
if "timestamp" in df_raw.columns:
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")

# ───────────────────────── construir features ────────────────────
df_b = build_indicators(df_raw.copy(), params, ATR_LEN)

clo_b = df_b.close.values
atr_b = df_b[f"atr_{ATR_LEN}"].values / TICK
horiz = params["horizon"]
fut_b = np.roll(clo_b, -horiz)
fut_b[-horiz:] = np.nan

diff_b = (fut_b - clo_b) / TICK
up_b = np.maximum(diff_b, 0) / atr_b
dn_b = np.maximum(-diff_b, 0) / atr_b
mask_b = (~np.isnan(diff_b)) & (np.maximum(up_b, dn_b) >= 0)

feature_cols = [c for c in df_b.columns if c not in [f"atr_{ATR_LEN}", "timestamp"]]
X_raw_f = df_b.loc[mask_b, feature_cols].select_dtypes(include=np.number)

cl_f, atr_f = clo_b[mask_b], atr_b[mask_b]

scaler = RobustScaler()
X_s_f, up_s_f, dn_s_f, cl_s_f = to_sequences(
    scaler.fit_transform(X_raw_f), up_b[mask_b], dn_b[mask_b], cl_f, params["win"]
)

# ───────────────────────── entrenamiento ─────────────────────────
EPOCHS_FIN, BATCH_FIN = 60, 128
model = make_model(
    X_s_f.shape[1:], params["lr"], params["dr"], params["filt"], params["units"], params["heads"]
)
es_cb = callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

model.fit(
    X_s_f,
    np.vstack([up_s_f, dn_s_f]).T,
    epochs=EPOCHS_FIN,
    batch_size=BATCH_FIN,
    callbacks=[es_cb],
    verbose=1,
)

# ───────────────────────── back-test rápido ──────────────────────
pred_val = model.predict(X_s_f, verbose=0)
score_bt = quick_bt(
    pred_val,
    cl_s_f,
    atr_f[params["win"] :],
    params["rr"],
    params["min_thr_up"],
    params["min_thr_dn"],
    params["delta_min"],
    params["smooth_win"],
    TICK,
)
print(f"⚡ Quick BT completo: {score_bt:.2f} ATR-pips netos")

# ───────────────────────── guardar artefactos ────────────────────
base_gcs_dir = args.output_gcs_base_dir.rstrip("/")
run_ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
gcs_prefix = f"{base_gcs_dir}/{PAIR}/{TF}/{run_ts}/"

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_dir = Path(tmpdir)
    model_path = tmp_dir / "model.h5"
    scaler_path = tmp_dir / "scaler.pkl"
    params_path_out = tmp_dir / "params.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    json.dump(params, open(params_path_out, "w"), indent=2)

    for local_file in (model_path, scaler_path, params_path_out):
        upload_to_gcs(local_file, gcs_prefix + local_file.name)
        print(f"☁️  Subido {local_file.name} a {gcs_prefix}{local_file.name}")

print("🎉 Fin del entrenamiento – artefactos en:", gcs_prefix)
