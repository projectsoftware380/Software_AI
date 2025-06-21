# src/components/train_lstm/task.py
"""
Tarea de Entrenamiento del Modelo LSTM.

Este script se ejecuta DENTRO de un CustomJob de Vertex AI en una máquina con GPU.
NO es un componente de KFP directamente.

Responsabilidades:
1.  Recibir los datos de entrada y los archivos de parámetros optimizados.
2.  Preprocesar los datos y crear secuencias para el LSTM.
3.  Construir el modelo Keras con la arquitectura especificada.
4.  Entrenar el modelo.
5.  Guardar el modelo entrenado y el scaler en la ruta GCS de salida especificada.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, optimizers

from src.shared import constants, gcs_utils, indicators

# --- Configuración (Sin Cambios) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# --- Helpers de Modelo y Secuencias (Sin Cambios) ---
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

def to_sequences(mat, up, dn, win):
    X, y_up, y_dn = [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i - win : i])
        y_up.append(up[i])
        y_dn.append(dn[i])
    return np.asarray(X, np.float32), np.asarray(y_up, np.float32), np.asarray(y_dn, np.float32)

# --- Función Principal de la Tarea (Ajustada) ---
def run_lstm_training(
    pair: str,
    timeframe: str,
    params_file: str,
    features_gcs_path: str,
    output_gcs_dir: str,  # <-- AJUSTE: Recibe el directorio de salida final
):
    """
    Orquesta el proceso completo de entrenamiento del modelo LSTM.
    """
    logger.info(f"--- Iniciando entrenamiento LSTM para el par: {pair} ---")

    # Verificar que estamos en un entorno con GPU
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU requerida y no detectada — abortando entrenamiento.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    logger.info(f"🚀 {len(gpus)} GPU(s) detectadas y configuradas.")

    # Cargar datos y parámetros
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        local_features_path = gcs_utils.download_gcs_file(features_gcs_path, tmp_path)
        df_full = pd.read_parquet(local_features_path)
        
        local_params_path = gcs_utils.download_gcs_file(params_file, tmp_path)
        with open(local_params_path) as f:
            p = json.load(f)

        # Preprocesamiento de datos
        df_ind = indicators.build_indicators(df_full.copy(), p, atr_len=14)
        
        tick = 0.01 if pair.endswith("JPY") else 0.0001
        horizon = p.get("win", 20)
        
        closes = df_ind.close.values
        atr_vals = df_ind[f"atr_14"].values / tick
        
        future_prices = np.roll(closes, -horizon)
        future_prices[-horizon:] = np.nan
        price_diffs = (future_prices - closes) / tick
        
        up_targets = np.maximum(price_diffs, 0) / atr_vals
        dn_targets = np.maximum(-price_diffs, 0) / atr_vals
        
        valid_mask = ~np.isnan(price_diffs)
        
        feature_cols = [c for c in df_ind.columns if "atr_" not in c and c != "timestamp"]
        X_data = df_ind.loc[valid_mask, feature_cols].select_dtypes(include=np.number).astype(np.float32)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        X_seq, y_up_seq, y_dn_seq = to_sequences(
            X_scaled, up_targets[valid_mask], dn_targets[valid_mask], p["win"]
        )
        
        if len(X_seq) == 0:
            raise ValueError("No se generaron secuencias, revisa los parámetros y los datos.")

        X_train, X_val, y_up_train, y_up_val, y_dn_train, y_dn_val = train_test_split(
            X_seq, y_up_seq, y_dn_seq, test_size=0.2, shuffle=False
        )

        # Construir y entrenar el modelo
        model = make_model(X_train.shape[1:], p["lr"], p["dr"], p["filt"], p["units"], p["heads"])
        
        model.fit(
            X_train, np.vstack((y_up_train, y_dn_train)).T,
            validation_data=(X_val, np.vstack((y_up_val, y_dn_val)).T),
            epochs=100,
            batch_size=128,
            verbose=2,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        # Guardar artefactos
        logger.info(f"Guardando artefactos del modelo en: {output_gcs_dir}")
        
        # Guardar modelo Keras
        model_uri = f"{output_gcs_dir}/model.keras"
        model.save(model_uri)
        gcs_utils.verify_gcs_file_exists(model_uri) # Verificación
        
        # Guardar scaler
        scaler_uri = f"{output_gcs_dir}/scaler.pkl"
        with tempfile.NamedTemporaryFile() as tmp_scaler:
            joblib.dump(scaler, tmp_scaler.name)
            gcs_utils.upload_gcs_file(Path(tmp_scaler.name), scaler_uri)
        gcs_utils.verify_gcs_file_exists(scaler_uri) # Verificación

        # Guardar los parámetros usados
        params_uri = f"{output_gcs_dir}/params.json"
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp_params:
            json.dump(p, tmp_params, indent=2)
            tmp_params_path = tmp_params.name
        gcs_utils.upload_gcs_file(Path(tmp_params_path), params_uri)
        gcs_utils.verify_gcs_file_exists(params_uri) # Verificación
        os.remove(tmp_params_path)

        logger.info(f"✅ Entrenamiento y guardado para {pair} completado con éxito.")


# --- Punto de Entrada para Ejecución como Script (Ajustado) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tarea de Entrenamiento de Modelo LSTM en Vertex AI.")
    
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--features-gcs-path", required=True)
    parser.add_argument("--output-gcs-dir", required=True) # <-- AJUSTE: Recibe el dir final
    
    args = parser.parse_args()
    
    run_lstm_training(
        pair=args.pair,
        timeframe=args.timeframe,
        params_file=args.params_file,
        features_gcs_path=args.features_gcs_path,
        output_gcs_dir=args.output_gcs_dir,
    )