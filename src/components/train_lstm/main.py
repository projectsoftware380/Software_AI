# RUTA: src/components/train_lstm/main.py
# CÓDIGO ACTUALIZADO – fuerza presencia de GPU

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from google.cloud import storage
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, mixed_precision

# ──────────────────────── logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ───────── reproducibilidad + GPU check (fail-fast) ─────────
def setup_environment() -> None:
    """Configura semillas y obliga a que exista GPU."""
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        # ‼️ Abortamos temprano si Vertex arrancó en CPU
        raise RuntimeError("GPU requerida y no detectada — abortando entrenamiento.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy("mixed_float16")
    logger.info("🚀 GPU(s) detectadas: %s | mixed_precision: %s",
                [g.name for g in gpus], mixed_precision.global_policy().name)

# ────────────── helpers de indicadores y secuencias ─────────
def build_indicators(df, sma_len, rsi_len, macf, macs, stoch_len, atr_len):
    out = df.copy()
    out[f"sma_{sma_len}"] = ta.sma(out.close, length=sma_len)
    out[f"rsi_{rsi_len}"] = ta.rsi(out.close, length=rsi_len)
    macd = ta.macd(out.close, fast=macf, slow=macs, signal=9)
    out[[f"macd_{macf}_{macs}", f"macd_signal_{macf}_{macs}", f"macd_hist_{macf}_{macs}"]] = macd
    stoch = ta.stoch(out.high, out.low, out.close, k=stoch_len, d=3)
    out[[f"stoch_k_{stoch_len}", f"stoch_d_{stoch_len}"]] = stoch
    out[f"atr_{atr_len}"] = ta.atr(out.high, out.low, out.close, length=atr_len)
    out.bfill(inplace=True)
    return out

def to_sequences(mat, up, dn, win):
    X, y_up, y_dn = [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i - win : i])
        y_up.append(up[i])
        y_dn.append(dn[i])
    return (
        np.asarray(X, np.float32),
        np.asarray(y_up, np.float32),
        np.asarray(y_dn, np.float32),
    )

# ───────────────── modelo LSTM ──────────────────────────────
def make_model(inp_sh, lr, dr, filt, units, heads):
    inp = layers.Input(shape=inp_sh, dtype=tf.float32)
    x = layers.Conv1D(filt, 3, padding="same", activation="relu")(inp)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mae")
    return model

# ──────────── helpers de GCS (sin cambios) ────────────────
def download_gcs_file(gcs_uri: str, local_path: str | Path):
    logger.info("Descargando %s → %s", gcs_uri, local_path)
    client = storage.Client()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(local_path)

def upload_local_directory_to_gcs(local_path: str | Path, gcs_uri: str):
    logger.info("Subiendo %s → %s", local_path, gcs_uri)
    client = storage.Client()
    bucket_name, dest_prefix = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    for local_file in Path(local_path).rglob("*"):
        if local_file.is_file():
            remote_path = Path(dest_prefix) / local_file.relative_to(local_path)
            bucket.blob(str(remote_path)).upload_from_filename(str(local_file))

# ─────────────────── flujo principal ───────────────────────
def train_final_model(
    pair: str,
    timeframe: str,
    params_path: str,
    features_gcs_path: str,
    output_gcs_base_dir: str,
):
    setup_environment()

    tmp = Path("/tmp/data")
    tmp.mkdir(exist_ok=True)

    local_params = tmp / "best_params.json"
    download_gcs_file(params_path, local_params)
    best_params = json.loads(local_params.read_text())
    logger.info("Hiperparámetros cargados.")

    local_features = tmp / Path(features_gcs_path).name
    download_gcs_file(features_gcs_path, local_features)
    df_raw = pd.read_parquet(local_features).reset_index(drop=True)
    logger.info("Parquet de features cargado: shape %s", df_raw.shape)

    tick = 0.01 if pair.endswith("JPY") else 0.0001
    atr_len = 14
    df_b = build_indicators(
        df_raw,
        best_params["sma_len"],
        best_params["rsi_len"],
        best_params["macd_fast"],
        best_params["macd_slow"],
        best_params["stoch_len"],
        atr_len,
    )

    clo_b = df_b.close.values
    atr_b = df_b[f"atr_{atr_len}"].values / tick
    horiz = best_params["horizon"]

    fut_b = np.roll(clo_b, -horiz)
    fut_b[-horiz:] = np.nan
    diff_b = (fut_b - clo_b) / tick
    up_b = np.maximum(diff_b, 0) / atr_b
    dn_b = np.maximum(-diff_b, 0) / atr_b
    mask_b = (~np.isnan(diff_b)) & (np.maximum(up_b, dn_b) >= 0)

    cols_exclude = [c for c in df_b.columns if "atr_" in c] + ["timestamp"]
    feature_cols = df_b.columns.difference(cols_exclude)
    X_raw_f = (
        df_b.loc[mask_b, feature_cols]
        .select_dtypes(include=np.number)
        .astype(np.float32)
    )

    y_up_f, y_dn_f = up_b[mask_b], dn_b[mask_b]
    scaler_final = RobustScaler()
    X_scaled = scaler_final.fit_transform(X_raw_f)
    X_seq, y_up_seq, y_dn_seq = to_sequences(X_scaled, y_up_f, y_dn_f, best_params["win"])
    logger.info("Datos listos para entrenamiento: X %s", X_seq.shape)

    model_final = make_model(
        X_seq.shape[1:],
        best_params["lr"],
        best_params["dr"],
        best_params["filt"],
        best_params["units"],
        best_params["heads"],
    )
    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    logger.info("🏋️  Entrenando LSTM final…")
    model_final.fit(
        X_seq,
        np.vstack([y_up_seq, y_dn_seq]).T,
        epochs=60,
        batch_size=128,
        verbose=1,
        callbacks=[es],
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    gcs_out = f"{output_gcs_base_dir}/{pair}/{timeframe}/{ts}"
    local_artifacts = Path(f"/tmp/artifacts/{ts}"); local_artifacts.mkdir(parents=True)

    model_final.save(local_artifacts / "model.h5")
    joblib.dump(scaler_final, local_artifacts / "scaler.pkl")
    (local_artifacts / "params.json").write_text(json.dumps(best_params, indent=4))

    upload_local_directory_to_gcs(local_artifacts, gcs_out)
    logger.info("✅ Artefactos subidos a %s", gcs_out)

# ───────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Entrenador de LSTM Vertex AI")
    ap.add_argument("--params", required=True)
    ap.add_argument("--output-gcs-base-dir", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--features-gcs-path", required=True)
    a = ap.parse_args()

    train_final_model(
        pair=a.pair,
        timeframe=a.timeframe,
        params_path=a.params,
        features_gcs_path=a.features_gcs_path,
        output_gcs_base_dir=a.output_gcs_base_dir,
    )
