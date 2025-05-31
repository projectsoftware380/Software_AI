#!/usr/bin/env python3
"""
Entrena el modelo final usando los hiperparámetros de un JSON (local o gs://).
Sube los artefactos a GCS con versionado por timestamp.
"""

# ───────────────────── imports estándar / utilidades ───────────
import os, sys, json, gc, random, warnings, tempfile
from pathlib import Path
import argparse
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
np.random.seed(42)

import pandas as pd, joblib
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.preprocessing import RobustScaler
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision

# 👉 NUEVO: función centralizada de indicadores
# Importación corregida para módulos de core
from indicators import build_indicators

# reproducibilidad y configuración GPU
tf.random.set_seed(42); random.seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
mixed_precision.set_global_policy("mixed_float16")

ATR_LEN = 14

# ───────────────────── funciones reutilizables ────────────────
def to_sequences(mat, up, dn, closes, win):
    X, y_up, y_dn, cl = [], [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i - win : i])
        y_up.append(up[i]); y_dn.append(dn[i]); cl.append(closes[i])
    return (
        np.asarray(X, np.float32),
        np.asarray(y_up, np.float32),
        np.asarray(y_dn, np.float32),
        np.asarray(cl, np.float32),
    )

def make_model(inp_sh, lr, dr, filt, units, heads):
    inp = layers.Input(shape=inp_sh, dtype=tf.float32)
    x   = layers.Conv1D(filt, 3, padding="same", activation="relu")(inp)
    x   = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x   = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizers.Adam(lr), loss="mae")
    return model

def quick_bt(pred, closes, atr_pips, rr, up_thr, dn_thr, delta_min, smooth_win, tick):
    net, pos = 0.0, False
    dq = deque(maxlen=smooth_win)
    for (u, d), price, atr in zip(pred, closes, atr_pips):
        mag, diff = max(u, d), abs(u - d)
        raw = 1 if u > d else -1
        dq.append(raw if mag >= (up_thr if raw == 1 else dn_thr) and diff >= delta_min else 0)
        buy_cnt, sell_cnt = dq.count(1), dq.count(-1)
        signal = 1 if buy_cnt > smooth_win // 2 else -1 if sell_cnt > smooth_win // 2 else 0

        if not pos and signal:
            pos, entry_price, entry_dir = True, price, signal
            sl = (up_thr if signal == 1 else dn_thr) * atr
            tp = rr * sl
            continue

        if pos and signal:
            cand_sl = (up_thr if signal == 1 else dn_thr) * atr
            cand_tp = rr * cand_sl
            sl = min(sl, cand_sl); tp = max(tp, cand_tp)

        if pos:
            pnl = (price - entry_price) / tick if entry_dir == 1 else (entry_price - price) / tick
            if pnl >= tp or pnl <= -sl:
                net += tp if pnl >= tp else -sl
                pos = False
    return net

# ───────────────────── helpers GCS ─────────────────────────────
def gcs_client():
    """
    Si GOOGLE_APPLICATION_CREDENTIALS está definida, utiliza esas credenciales
    para acceder a Google Cloud Storage.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        return storage.Client(credentials=creds)
    return storage.Client()

def download_from_gcs(gcs_uri: str) -> Path:
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    local_tmp = Path(tempfile.mkdtemp()) / Path(blob_name).name
    gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(local_tmp)
    return local_tmp

def upload_to_gcs(local_path: Path, gcs_uri: str):
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    gcs_client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))

# ───────────────────── CLI ─────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--params", required=True,
                    help="Ruta local o gs://... del JSON con hiperparámetros (salida de optimize_lstm.py).")
# --upload ya no es necesario como argumento CLI, siempre se subirá a GCS en GCP.
# El script simplemente guardará los artefactos en la ruta GCS determinada.
args = parser.parse_args()

# ───────────────────── Cargar JSON de Hiperparámetros ─────────────────────────
params_path = Path(args.params) if not args.params.startswith("gs://") else download_from_gcs(args.params)
payload = json.loads(params_path.read_text())
PAIR, TF = payload["pair"], payload["timeframe"]
tick = 0.01 if PAIR.endswith("JPY") else 0.0001

# ───────────────────── Cargar features (los 5 años de datos de optimización) ──
# feat_path provendrá del 'features_path' guardado en el JSON de hiperparámetros
# que optimize_lstm.py generó (apuntando a los 5 años de datos filtrados).
feat_path  = payload["features_path"]
feat_local = Path(feat_path) if not feat_path.startswith("gs://") else download_from_gcs(feat_path)
df_raw     = pd.read_parquet(feat_local).reset_index(drop=True)

# Asegurarse de que la columna 'timestamp' sea de tipo datetime.
if 'timestamp' in df_raw.columns:
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms', errors='coerce')

# ───────────────────── Preparar datos finales para entrenamiento ──────────────
# Pasar una copia de df_raw a build_indicators.
df_b = build_indicators(df_raw.copy(), payload, ATR_LEN)

# Columnas necesarias para backtesting y secuencias:
clo_b = df_b.close.values
atr_b = df_b[f"atr_{ATR_LEN}"].values / tick
horiz = payload["horizon"]
fut_b = np.roll(clo_b, -horiz); fut_b[-horiz:] = np.nan
diff_b = (fut_b - clo_b) / tick
up_b   = np.maximum(diff_b, 0) / atr_b
dn_b   = np.maximum(-diff_b, 0) / atr_b
mask_b = (~np.isnan(diff_b)) & (np.maximum(up_b, dn_b) >= 0)

# Excluir explícitamente 'atr_LEN' y 'timestamp' de los features para el scaler.
features_for_scaler = [col for col in df_b.columns if col not in [f"atr_{ATR_LEN}", 'timestamp']]
X_raw_f = df_b.loc[mask_b, features_for_scaler]

# Asegurarse de que X_raw_f sea numérico.
X_raw_f = X_raw_f.select_dtypes(include=np.number)

cl_f, atr_f = clo_b[mask_b], atr_b[mask_b]

scaler_final = RobustScaler()
X_s_f, up_s_f, dn_s_f, cl_s_f = to_sequences(
    scaler_final.fit_transform(X_raw_f),
    up_b[mask_b], dn_b[mask_b], cl_f, payload["win"]
)

# ───────────────────── Entrenamiento final ─────────────────────
# Usar los epochs y batch_size definidos en el script, no de Optuna, si estos son fijos para el entrenamiento final.
# EPOCHS_FIN y BATCH_FIN se mantienen como constantes.
EPOCHS_FIN, BATCH_FIN = 60, 128
model_final = make_model(
    X_s_f.shape[1:], payload["lr"], payload["dr"],
    payload["filt"], payload["units"], payload["heads"]
)
es_final = callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
model_final.fit(
    X_s_f, np.vstack([up_s_f, dn_s_f]).T,
    epochs=EPOCHS_FIN, batch_size=BATCH_FIN,
    verbose=1, callbacks=[es_final]
)

# ───────────────────── Back-test rápido (logging) ───────────────
pred_val = model_final.predict(X_s_f, verbose=0)
score = quick_bt(
    pred_val, cl_s_f, atr_f[payload["win"]:],
    payload["rr"], payload["min_thr_up"], payload["min_thr_dn"],
    payload["delta_min"], payload["smooth_win"], tick
)
print(f"⚡ Quick BT para entrenamiento completo: {score:.2f} ATR-pips netos")

# ───────────────────── Guardar artefactos en GCS con versionado por timestamp ──
# Generar un timestamp para el versionado de la carpeta
timestamp_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
# La ruta base para los modelos entrenados en GCS
GCS_MODELS_BASE_PATH = "gs://trading-ai-models-460823/models/LSTM" # Tu bucket de modelos
gcs_prefix = f"{GCS_MODELS_BASE_PATH}/{PAIR}/{TF}/{timestamp_str}/"

# Crear un directorio temporal local para guardar los artefactos antes de subir
with tempfile.TemporaryDirectory() as tmpdir:
    local_out = Path(tmpdir) / "artifacts" # Subdirectorio para los artefactos temporales
    local_out.mkdir(parents=True, exist_ok=True) # Asegurarse de que el subdirectorio exista

    # Guardar artefactos localmente en el directorio temporal
    model_final.save(local_out / "model.h5")
    joblib.dump(scaler_final, local_out / "scaler.pkl")
    # Es crucial guardar el payload original para reproducibilidad/uso posterior
    json.dump(payload, open(local_out / "params.json", "w"), indent=2)

    print(f"✅ Artefactos guardados localmente en {local_out}")

    # Subir artefactos a GCS
    for fname in ["model.h5", "scaler.pkl", "params.json"]:
        upload_to_gcs(local_out / fname, gcs_prefix + fname)
        print(f"☁️  Subido {fname} a {gcs_prefix}{fname}")

print("🎉 Fin del entrenamiento final", datetime.utcnow().isoformat(), "UTC")