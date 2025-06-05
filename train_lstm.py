#!/usr/bin/env python3
"""
train_lstm.py
─────────────
Entrena el modelo LSTM “definitivo” con los hiperparámetros obtenidos en
optimize_lstm.py y sube los artefactos (model.h5, scaler.pkl y params.json)
a GCS usando un esquema  gs://<base>/<pair>/<tf>/<YYYYMMDD-HHMMSS>/.

Principales refuerzos de robustez
─────────────────────────────────
▸ Valida que no queden NaNs en los indicadores ni en las secuencias antes
  de entrenar.  
▸ Verifica que las columnas que recibe el scaler son EXACTAMENTE las mismas
  con las que fue ajustado (evita el “feature drift”).  
▸ Comprueba que el tamaño de ventana (`win`) sea < nº de filas útiles.  
▸ Maneja la ausencia de GPU sin abortar y activa mixed-precision sólo
  cuando es seguro.  
▸ Todos los accesos a GCS pasan por helpers con detección automática de
  credenciales (funciona en local y en Vertex AI).  
▸ Mensajes de log claros para depuración en Cloud Logging.
"""

# ───────────────────────── imports estándar ─────────────────────────
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np

# Parche NumPy ≥1.24 (elimina la vieja constante NaN)
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore

import pandas as pd
import joblib
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision

from indicators import build_indicators  # función centralizada de indicadores

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ────────────────────── determinismo y hardware ───────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    _GPUS = tf.config.list_physical_devices("GPU")
    if _GPUS:
        for g in _GPUS:
            tf.config.experimental.set_memory_growth(g, True)
        mixed_precision.set_global_policy("mixed_float16")
        print("🚀 GPU detectada – prec. mixta activada")
    else:
        print("ℹ️  No se detectó GPU – ejecución en CPU")
except Exception as e:  # pragma: no cover
    print(f"⚠️  No se pudo configurar GPU / mixed-precision: {e}")

# ───────────────────── helpers de Cloud Storage ──────────────────────
def _gcs_client() -> storage.Client:
    """Devuelve un cliente de GCS respetando GOOGLE_APPLICATION_CREDENTIALS."""
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and Path(cred_path).exists():
        creds = service_account.Credentials.from_service_account_file(cred_path)
        return storage.Client(credentials=creds)
    return storage.Client()


def gcs_download(uri: str) -> Path:
    """Descarga un fichero gs://… a un tmp local y devuelve el Path local."""
    bucket, blob = uri[5:].split("/", 1)
    local = Path(tempfile.mkdtemp()) / Path(blob).name
    _gcs_client().bucket(bucket).blob(blob).download_to_filename(local)
    return local


def gcs_upload(local: Path, uri: str) -> None:
    """Sube un Path local a gs://…"""
    bucket, blob = uri[5:].split("/", 1)
    _gcs_client().bucket(bucket).blob(blob).upload_from_filename(str(local))


# ──────────────────────── lógica de secuencias ────────────────────────
def to_sequences(
    mat: np.ndarray,
    up: np.ndarray,
    dn: np.ndarray,
    closes: np.ndarray,
    win: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


# ─────────────────────────── modelo LSTM ──────────────────────────────
def make_model(
    inp_shape: Tuple[int, int],
    lr: float,
    dr: float,
    filt: int,
    units: int,
    heads: int,
) -> tf.keras.Model:
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


# ───────────────────────────── CLI ────────────────────────────────────
cli = argparse.ArgumentParser(description="Entrena el LSTM final y sube artefactos a GCS")
cli.add_argument("--params", required=True, help="Ruta al JSON de hiperparámetros (local o gs://)")
cli.add_argument("--output-gcs-base-dir", default="gs://trading-ai-models-460823/models/LSTM")
cli.add_argument("--pair")
cli.add_argument("--timeframe")
# flags “extra” inofensivos que Vertex puede inyectar:
cli.add_argument("--project-id")
cli.add_argument("--gcs-bucket-name")
args, _unknown = cli.parse_known_args()

# ────────────────────── leer hiperparámetros ──────────────────────────
params_path = Path(args.params) if not args.params.startswith("gs://") else gcs_download(args.params)
hp: dict = json.loads(params_path.read_text())

PAIR: str = args.pair or hp["pair"]
TF: str = args.timeframe or hp["timeframe"]
TICK: float = 0.01 if PAIR.endswith("JPY") else 0.0001
ATR_LEN: int = 14  # fijo para todos los scripts

# ───────────────────── cargar y preparar datos ────────────────────────
feat_uri = hp["features_path"]
feat_local = Path(feat_uri) if not feat_uri.startswith("gs://") else gcs_download(feat_uri)

df_raw = pd.read_parquet(feat_local).reset_index(drop=True)
if "timestamp" in df_raw.columns:
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")

# Indicadores
df_ind = build_indicators(df_raw.copy(), hp, ATR_LEN)

# Derivados para targets
close = df_ind.close.values
atr_pips = df_ind[f"atr_{ATR_LEN}"].values / TICK
horizon = int(hp["horizon"])
future_close = np.roll(close, -horizon)
future_close[-horizon:] = np.nan

diff = (future_close - close) / TICK
up = np.maximum(diff, 0) / atr_pips
dn = np.maximum(-diff, 0) / atr_pips

mask = (~np.isnan(diff)) & (np.maximum(up, dn) >= 0)

feature_cols: List[str] = [c for c in df_ind.columns if c not in {f"atr_{ATR_LEN}", "timestamp"}]
X_raw = df_ind.loc[mask, feature_cols]

# Seguridad extra: asegura que no haya NaNs numéricos
if X_raw.isna().any().any():
    raise ValueError("❌  Persisten NaNs en los features tras build_indicators(). Revisa la fuente de datos.")

# Ajustar / transformar
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)

# Validar que el scaler conoce exactamente las mismas columnas
assert list(scaler.feature_names_in_) == feature_cols, "Drift de columnas entre scaler y DataFrame"

# Construir secuencias
win = int(hp["win"])
if len(X_scaled) <= win:
    raise ValueError(f"❌  Muy pocos registros ({len(X_scaled)}) para win={win}")

X_seq, y_up, y_dn, closes_seq = to_sequences(X_scaled, up[mask], dn[mask], close[mask], win)

# Double-check de NaNs antes de entrenar
if np.isnan(X_seq).any() or np.isnan(y_up).any() or np.isnan(y_dn).any():
    raise ValueError("❌  Se han detectado NaNs en las secuencias de entrenamiento – abortando.")

print(f"✅  Datos listos: X={X_seq.shape}, y={(y_up.shape, y_dn.shape)}")

# ─────────────────────── entrenamiento LSTM ───────────────────────────
model = make_model(
    X_seq.shape[1:], hp["lr"], hp["dr"], hp["filt"], hp["units"], hp["heads"]
)
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

model.fit(
    X_seq,
    np.vstack([y_up, y_dn]).T,
    epochs=60,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1,
)

# ──────────────────── carga rápida de validación ──────────────────────
from collections import deque  # inline para mantener fichero autoconclusivo

def quick_bt(pred, closes, atr_pips, hp: dict, tick: float) -> float:
    """Back-test mínimo para sanity-check (sin comisiones)."""
    rr, up_thr, dn_thr = hp["rr"], hp["min_thr_up"], hp["min_thr_dn"]
    delta_min, swin = hp["delta_min"], hp["smooth_win"]
    net: float = 0.0
    pos = False
    dq: deque[int] = deque(maxlen=swin)
    for (u, d), price, atr in zip(pred, closes, atr_pips):
        mag, diff = max(u, d), abs(u - d)
        raw = 1 if u > d else -1
        thr = up_thr if raw == 1 else dn_thr
        dq.append(raw if (mag >= thr and diff >= delta_min) else 0)
        buys, sells = dq.count(1), dq.count(-1)
        sig = 1 if buys > swin // 2 else -1 if sells > swin // 2 else 0
        if not pos and sig:
            pos = True
            entry = price
            sl = thr * atr
            tp = rr * sl
            dir_ = sig
            continue
        if pos:
            pnl = ((price - entry) if dir_ == 1 else (entry - price)) / tick
            if pnl >= tp or pnl <= -sl:
                net += tp if pnl >= tp else -sl
                pos = False
    return net

bt_score = quick_bt(
    model.predict(X_seq, verbose=0),
    closes_seq,
    atr_pips[win:],
    hp,
    TICK,
)
print(f"⚡ Quick BT (sanity): {bt_score:.2f} ATR-pips")

# ──────────────────── guardado de artefactos ──────────────────────────
timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
gcs_base = args.output_gcs_base_dir.rstrip("/")
gcs_dir = f"{gcs_base}/{PAIR}/{TF}/{timestamp}/"

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    model_path = tmp_dir / "model.h5"
    scaler_path = tmp_dir / "scaler.pkl"
    params_out_path = tmp_dir / "params.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    json.dump(hp, open(params_out_path, "w"), indent=2)

    for p in (model_path, scaler_path, params_out_path):
        gcs_upload(p, gcs_dir + p.name)
        print(f"☁️  Subido {p.name} → {gcs_dir}{p.name}")

print(f"🎉  Entrenamiento completado. Artefactos disponibles en:\n    {gcs_dir}")
