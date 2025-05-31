#!/usr/bin/env python3
"""
prepare_rl_data.py
──────────────────
Genera la matriz de observaciones y señales crudas que usará el agente PPO.

Entrada (argumentos CLI)
-----------------------
--model      Ruta gs:// al modelo LSTM (.h5)
--scaler     Ruta gs:// al scaler (.pkl)
--features   Ruta gs:// al .parquet con features OHLC
--params     Ruta gs:// al .json con hiperparámetros
--output     Ruta gs:// donde se guardar? el archivo .npz de datos RL.

Salida
------
Un archivo .npz con:
    obs   : np.ndarray  [n_samples, 2 + embedding_dim]
    raw   : np.ndarray  [n_samples,]  (1 buy, -1 sell, 0 no-trade)
    closes: np.ndarray  [n_samples,]  (precio de cierre para simulaci?n)
"""

# ────────── 1. imports principales (sin 'ensure') ──────────────────────────
import os, sys, json, warnings, random, tempfile
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd, joblib
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras import models, mixed_precision
from collections import deque

# 👉 NUEVO: funci?n centralizada de indicadores
# Importaci?n corregida para m?dulos de core
from indicators import build_indicators

warnings.filterwarnings("ignore", category=FutureWarning)
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
mixed_precision.set_global_policy("mixed_float16")

# ────────── 2. helpers GCS (sin cambios) ───────────────────────────────────
def gcs_client():
    """
    Si GOOGLE_APPLICATION_CREDENTIALS est? definida, utiliza esas credenciales
    para acceder a Google Cloud Storage.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        return storage.Client(credentials=creds)
    return storage.Client()

def download_gs(uri: str) -> Path:
    """
    Descarga un archivo desde GCS a un directorio temporal.
    """
    bucket, blob = uri[5:].split("/", 1)
    local = Path(tempfile.mkdtemp()) / Path(blob).name
    gcs_client().bucket(bucket).blob(blob).download_to_filename(local)
    return local

def upload_gs(local: Path, uri: str):
    """
    Subir archivo a GCS desde una ruta local.
    """
    bucket, blob = uri[5:].split("/", 1)
    gcs_client().bucket(bucket).blob(blob).upload_from_filename(str(local))

def maybe_local(path_or_uri: str) -> Path:
    """
    Verifica si la ruta es GCS o local, y descarga desde GCS si es necesario.
    """
    return download_gs(path_or_uri) if path_or_uri.startswith("gs://") else Path(path_or_uri)

# ────────── 3. CLI (ajustado para output) ──────────────────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("--model",    required=True, help="Ruta gs:// al modelo LSTM (.h5).")
cli.add_argument("--scaler",   required=True, help="Ruta gs:// al scaler (.pkl).")
cli.add_argument("--features", required=True, help="Ruta gs:// al .parquet con features OHLC.")
cli.add_argument("--params",   required=True, help="Ruta gs:// al .json con hiperpar?metros del LSTM.")
cli.add_argument("--output",   required=True, help="Ruta gs:// donde se guardar? el archivo .npz de datos RL.")
args = cli.parse_args()

# ────────── 4. cargar artefactos (sin cambios) ───────────────────────────
model_path    = maybe_local(args.model)
scaler_path   = maybe_local(args.scaler)
features_path = maybe_local(args.features)
params_path   = maybe_local(args.params)

hp   = json.loads(params_path.read_text())
PAIR = hp["pair"]; TF = hp["timeframe"]

tick = 0.01 if PAIR.endswith("JPY") else 0.0001
ATR_LEN = 14

# modelo y embedding
lstm_model = tf.keras.models.load_model(model_path, compile=False)
emb_model  = models.Model(lstm_model.input, lstm_model.layers[-2].output)

# scaler
scaler = joblib.load(scaler_path)

# ────────── 5. preparar DataFrame con indicadores ────────────────────────
df_raw = pd.read_parquet(features_path).reset_index(drop=True)

# Asegurar que la columna 'timestamp' sea de tipo datetime y excluirla del escalado.
if 'timestamp' in df_raw.columns:
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms', errors='coerce')

# Pasa una copia de df_raw a build_indicators, que espera 'open', 'high', 'low', 'close'
df = build_indicators(df_raw.copy(), hp, ATR_LEN)

# ────────── 6. escalar y crear secuencias ───────────────────────
def make_seq(arr: np.ndarray, w: int) -> np.ndarray:
    """Crea ventanas [i-w : i) a lo largo de arr."""
    return np.stack([arr[i - w : i] for i in range(w, len(arr))]).astype(np.float32)

# Asegurarse de que las columnas pasadas al scaler sean las mismas que las usadas en el entrenamiento del scaler.
# Esto es crucial. El scaler.feature_names_in_ contiene los nombres de las columnas a las que se ajust? el scaler.
# Se asume que df (despu?s de build_indicators) contiene estas columnas.
if hasattr(scaler, 'feature_names_in_'):
    features_for_scaler_transform = [col for col in scaler.feature_names_in_ if col in df.columns]
    if not features_for_scaler_transform:
        logger.critical(f"Error: Ninguna columna de df coincide con las feature_names_in_ del scaler. Revise la consistencia.")
        sys.exit(1)
    X_scaled_df_input = df[features_for_scaler_transform]
else:
    # Fallback si el scaler no tiene feature_names_in_ (menos robusto, pero para compatibilidad si el scaler es antiguo)
    logger.warning("Scaler no tiene 'feature_names_in_'. Intentando escalar todas las columnas num?ricas excepto 'timestamp' y 'atr_LEN'.")
    features_for_scaler_transform = [col for col in df.columns if col not in ['timestamp', f"atr_{ATR_LEN}"] and pd.api.types.is_numeric_dtype(df[col])]
    X_scaled_df_input = df[features_for_scaler_transform]

# Asegurarse de que las columnas seleccionadas sean num?ricas
X_scaled_df_input = X_scaled_df_input.select_dtypes(include=np.number)
if X_scaled_df_input.empty:
    logger.critical(f"Error: El DataFrame de entrada para el scaler est? vac?o despu?s de la selecci?n de columnas num?ricas. Abortando.")
    sys.exit(1)


X_scaled    = scaler.transform(X_scaled_df_input)
X_seq       = make_seq(X_scaled, hp["win"])

# Asegurarse de que `closes` y `atr_pips` se obtengan de `df` despu?s de `build_indicators`
closes      = df.close.values[hp["win"]:]
atr_pips    = df[f"atr_{ATR_LEN}"].values[hp["win"]:] / tick

# ────────── 7. predicciones + embedding (sin cambios) ────────────────────
with tf.device("/GPU:0" if gpus else "/CPU:0"):
    preds = lstm_model.predict(X_seq, verbose=0)
    embs  = emb_model.predict(X_seq, verbose=0).astype(np.float32)

pred_up, pred_dn = preds[:, 0], preds[:, 1]
OBS = np.hstack([preds, embs]).astype(np.float32)   # (n, 2 + emb_dim)

# ────────── 8. generar raw_signal con l?gica PPO (sin cambios) ───────────
raw_signal = np.zeros(len(closes), dtype=np.int8)
dq = deque(maxlen=hp["smooth_win"])

for i, (up, dn, atr) in enumerate(zip(pred_up, pred_dn, atr_pips)):
    mag, diff = max(up, dn), abs(up - dn)
    raw_dir   = 1 if up > dn else -1
    condition = ((raw_dir == 1 and mag >= hp["min_thr_up"]) or
                 (raw_dir == -1 and mag >= hp["min_thr_dn"])) and diff >= hp["delta_min"]
    dq.append(raw_dir if condition else 0)

    buys, sells = dq.count(1), dq.count(-1)
    signal = 1 if buys  > hp["smooth_win"] // 2 else \
            -1 if sells > hp["smooth_win"] // 2 else 0
    raw_signal[i] = signal

# ────────── 9. guardar archivo .npz (ajustado para GCS) ─────────────────────────
# Crear un archivo temporal para guardar el NPZ antes de subirlo.
with tempfile.TemporaryDirectory() as tmpdir:
    local_tmp_file = Path(tmpdir) / "ppo_input_data.npz"
    np.savez(local_tmp_file, obs=OBS, raw=raw_signal, closes=closes)
    logger.info(f"Archivo .npz generado localmente: {local_tmp_file}")

    # Subir a GCS (args.output ahora ser? siempre una ruta gs://)
    upload_gs(local_tmp_file, args.output)
    logger.info(f"☁️  Subido a {args.output}")

logger.info("🎉 prepare_rl_data.py finalizado ?", datetime.utcnow().isoformat(), "UTC")