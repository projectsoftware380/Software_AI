# RUTA: src/components/train_lstm/main.py
# CÓDIGO CORREGIDO Y COMPLETO

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

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- Reproducibilidad y Configuración de GPU ---
def setup_environment():
    """Configura la semilla y optimizaciones de GPU para TensorFlow."""
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPUs habilitadas: %s", [g.name for g in gpus])
        mixed_precision.set_global_policy('mixed_float16')
        logger.info("Política de precisión mixta habilitada: %s", mixed_precision.global_policy().name)
    else:
        logger.warning("No se detectó GPU; se usará CPU.")

# --- Funciones de Preprocesamiento ---
def build_indicators(df, sma_len, rsi_len, macf, macs, stoch_len, atr_len):
    """Construye los indicadores técnicos sobre el DataFrame."""
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
    """Convierte los datos en secuencias para el LSTM."""
    X, y_up, y_dn = [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i-win:i])
        y_up.append(up[i])
        y_dn.append(dn[i])
    return (
        np.asarray(X, np.float32),
        np.asarray(y_up, np.float32),
        np.asarray(y_dn, np.float32),
    )

# --- Definición del Modelo ---
def make_model(inp_sh, lr, dr, filt, units, heads):
    """Crea y compila el modelo Keras."""
    inp = layers.Input(shape=inp_sh, dtype=tf.float32)
    x = layers.Conv1D(filt, 3, padding="same", activation="relu")(inp)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype='float32')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mae")
    return model

# --- Funciones de Interacción con GCS ---
def download_gcs_file(gcs_uri: str, local_path: str | Path):
    """Descarga un archivo desde GCS a una ruta local."""
    logger.info("Descargando de %s a %s", gcs_uri, local_path)
    client = storage.Client()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def upload_local_directory_to_gcs(local_path: str | Path, gcs_uri: str):
    """Sube el contenido de un directorio local a un prefijo en GCS."""
    logger.info("Subiendo directorio %s a %s", local_path, gcs_uri)
    client = storage.Client()
    bucket_name, dest_prefix = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    
    local_path = Path(local_path)
    for local_file in local_path.rglob('*'):
        if local_file.is_file():
            remote_path = Path(dest_prefix) / local_file.relative_to(local_path)
            blob = bucket.blob(str(remote_path))
            blob.upload_from_filename(str(local_file))

# --- Flujo Principal de Entrenamiento ---
def train_final_model(
    pair: str,
    timeframe: str,
    params_path: str,
    features_gcs_path: str,
    output_gcs_base_dir: str,
):
    """Función principal que ejecuta el pipeline de entrenamiento."""
    setup_environment()
    
    local_dir = Path("/tmp/data")
    local_dir.mkdir(exist_ok=True)
    
    local_params_path = local_dir / "best_params.json"
    download_gcs_file(params_path, local_params_path)
    with open(local_params_path) as f:
        best_params = json.load(f)
    logger.info("Hiperparámetros cargados.")
    
    local_features_path = local_dir / Path(features_gcs_path).name
    download_gcs_file(features_gcs_path, local_features_path)
    df_raw = pd.read_parquet(local_features_path).reset_index(drop=True)
    logger.info("Datos de características cargados desde '%s', shape: %s", features_gcs_path, df_raw.shape)

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
    
    # --- AJUSTE CLAVE: Excluir 'timestamp' junto con 'atr' ANTES de escalar ---
    cols_to_exclude = [col for col in df_b.columns if 'atr_' in col] + ['timestamp']
    feature_cols = df_b.columns.difference(cols_to_exclude)
    
    # ============================ INICIO DE LA CORRECCIÓN ============================
    # Se añade .select_dtypes(include=np.number) para asegurar que solo las columnas
    # numéricas se pasen al escalador, previniendo errores si alguna columna
    # no numérica (aparte de 'timestamp') estuviera presente.
    X_raw_f = df_b.loc[mask_b, feature_cols].select_dtypes(include=np.number)
    # ============================= FIN DE LA CORRECCIÓN ==============================
    
    y_up_f = up_b[mask_b]
    y_dn_f = dn_b[mask_b]
    
    scaler_final = RobustScaler()
    X_scaled = scaler_final.fit_transform(X_raw_f)
    X_seq, y_up_seq, y_dn_seq = to_sequences(X_scaled, y_up_f, y_dn_f, best_params["win"])
    
    logger.info("Datos preprocesados y listos para entrenamiento, shape X: %s", X_seq.shape)

    model_final = make_model(
        X_seq.shape[1:],
        best_params["lr"],
        best_params["dr"],
        best_params["filt"],
        best_params["units"],
        best_params["heads"],
    )
    model_final.summary()
    
    es_final = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    
    logger.info("Iniciando entrenamiento final...")
    model_final.fit(
        X_seq,
        np.vstack([y_up_seq, y_dn_seq]).T,
        epochs=60,
        batch_size=128,
        verbose=1,
        callbacks=[es_final],
    )
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    final_output_gcs_path = f"{output_gcs_base_dir}/{pair}/{timeframe}/{ts}"
    local_artifact_dir = Path(f"/tmp/artifacts/{ts}")
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Guardando artefactos en el directorio local: %s", local_artifact_dir)
    model_final.save(local_artifact_dir / "model.h5")
    joblib.dump(scaler_final, local_artifact_dir / "scaler.pkl")
    with open(local_artifact_dir / "params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    
    upload_local_directory_to_gcs(local_artifact_dir, final_output_gcs_path)
    logger.info("✅ Artefactos subidos exitosamente a: %s", final_output_gcs_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenador de Modelo LSTM para Vertex AI")
    parser.add_argument("--params", required=True, help="Ruta GCS al archivo de hiperparámetros JSON.")
    parser.add_argument("--output-gcs-base-dir", required=True, help="Directorio GCS base donde se guardarán los artefactos.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--features-gcs-path", required=True, help="Ruta GCS al archivo parquet de características.")

    args = parser.parse_args()

    train_final_model(
        pair=args.pair,
        timeframe=args.timeframe,
        params_path=args.params,
        features_gcs_path=args.features_gcs_path,
        output_gcs_base_dir=args.output_gcs_base_dir,
    )