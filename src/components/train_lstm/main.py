# src/components/train_lstm/main.py
"""
Tarea de Entrenamiento del Modelo LSTM. (Versión con Logging Robusto)

Este script se ejecuta DENTRO de un CustomJob de Vertex AI en una máquina con GPU.
NO es un componente de KFP directamente.

Responsabilidades:
1.  Recibir los datos de entrada y los archivos de parámetros optimizados.
2.  Preprocesar los datos y crear secuencias para el LSTM.
3.  Construir el modelo Keras con la arquitectura especificada.
4.  Entrenar el modelo.
5.  Guardar el modelo entrenado y sus artefactos en la ruta GCS de salida especificada.
6.  Limpiar las versiones antiguas de los modelos para mantener solo el más reciente.
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

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# --- Helpers de Modelo y Secuencias (Lógica Original Intacta) ---
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
        X.append(mat[i - win:i])
        y_up.append(up[i])
        y_dn.append(dn[i])
    return np.asarray(X, np.float32), np.asarray(y_up, np.float32), np.asarray(y_dn, np.float32)

# --- Función Principal de la Tarea ---
def run_lstm_training(
    pair: str,
    timeframe: str,
    params_file: str,
    features_gcs_path: str,
    output_gcs_dir: str,
    cleanup: bool = True,
):
    """
    Orquesta el proceso completo de entrenamiento del modelo LSTM.
    """
    # [LOG] Punto de control inicial.
    logger.info(f"▶️ Iniciando train_lstm para el par '{pair}' con los siguientes parámetros:")
    logger.info(f"  - Timeframe: {timeframe}")
    logger.info(f"  - Archivo de Parámetros: {params_file}")
    logger.info(f"  - Features GCS Path: {features_gcs_path}")
    logger.info(f"  - Directorio de Salida: {output_gcs_dir}")
    logger.info(f"  - Limpieza activada: {cleanup}")

    try:
        # [LOG] Verificación explícita del entorno de hardware.
        logger.info("Verificando disponibilidad de GPU...")
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("GPU requerida y no detectada — abortando entrenamiento.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info(f"✅ {len(gpus)} GPU(s) detectadas y configuradas exitosamente.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            logger.info("Descargando artefactos de entrada...")
            local_features_path = gcs_utils.download_gcs_file(features_gcs_path, tmp_path)
            local_params_path = gcs_utils.download_gcs_file(params_file, tmp_path)
            
            df_full = pd.read_parquet(local_features_path)
            with open(local_params_path) as f:
                p = json.load(f)

            logger.info(f"Artefactos cargados. Shape del DataFrame: {df_full.shape}. Parámetros: {p}")
            
            logger.info("Iniciando preprocesamiento de datos y creación de secuencias...")
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
                raise ValueError("No se generaron secuencias de entrenamiento. Revisa los parámetros 'win' y los datos de entrada.")
            
            logger.info(f"Creación de secuencias completada. Shape de X_seq: {X_seq.shape}")
            
            X_train, X_val, y_up_train, y_up_val, y_dn_train, y_dn_val = train_test_split(
                X_seq, y_up_seq, y_dn_seq, test_size=0.2, shuffle=False
            )
            logger.info(f"División de datos completada. Shape X_train: {X_train.shape}, Shape X_val: {X_val.shape}")
            
            model = make_model(X_train.shape[1:], p["lr"], p["dr"], p["filt"], p["units"], p["heads"])
            logger.info("Modelo LSTM construido. Iniciando entrenamiento (`model.fit`)...")
            
            model.fit(
                X_train, np.vstack((y_up_train, y_dn_train)).T,
                validation_data=(X_val, np.vstack((y_up_val, y_dn_val)).T),
                epochs=100,
                batch_size=128,
                verbose=2, # verbose=2 es bueno para logs de Vertex AI
                callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            )
            logger.info("Entrenamiento del modelo completado.")
            
            logger.info(f"Guardando artefactos del modelo en: {output_gcs_dir}")
            
            model_uri = f"{output_gcs_dir}/model.keras"
            model.save(model_uri)
            gcs_utils.verify_gcs_file_exists(model_uri)
            
            scaler_uri = f"{output_gcs_dir}/scaler.pkl"
            with tempfile.NamedTemporaryFile() as tmp_scaler:
                joblib.dump(scaler, tmp_scaler.name)
                gcs_utils.upload_gcs_file(Path(tmp_scaler.name), scaler_uri)
            gcs_utils.verify_gcs_file_exists(scaler_uri)

            params_uri = f"{output_gcs_dir}/params.json"
            with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp_params:
                json.dump(p, tmp_params, indent=2)
                tmp_params_path = tmp_params.name
            gcs_utils.upload_gcs_file(Path(tmp_params_path), params_uri)
            gcs_utils.verify_gcs_file_exists(params_uri)
            os.remove(tmp_params_path)
            
            logger.info("✅ Todos los artefactos del modelo han sido guardados y verificados.")

            if cleanup:
                base_cleanup_path = f"{constants.LSTM_MODELS_PATH}/{pair}/{timeframe}"
                logger.info(f"Iniciando limpieza de versiones antiguas de modelos en: {base_cleanup_path}")
                gcs_utils.keep_only_latest_version(base_cleanup_path)

    except Exception as e:
        # [LOG] Captura de error fatal.
        logger.critical(f"❌ Fallo fatal en el entrenamiento LSTM para el par '{pair}'. Error: {e}", exc_info=True)
        raise

    logger.info(f"🏁 Componente train_lstm para '{pair}' completado exitosamente.")


# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tarea de Entrenamiento de Modelo LSTM en Vertex AI.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--features-gcs-path", required=True)
    parser.add_argument("--output-gcs-dir", required=True)
    parser.add_argument("--cleanup", type=lambda x: (str(x).lower() == 'true'), default=True)
    
    args = parser.parse_args()

    # [LOG] Registro de los argumentos recibidos.
    logger.info("Componente 'train_lstm' iniciado con los siguientes argumentos:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
    
    run_lstm_training(
        pair=args.pair,
        timeframe=args.timeframe,
        params_file=args.params_file,
        features_gcs_path=args.features_gcs_path,
        output_gcs_dir=args.output_gcs_dir,
        cleanup=args.cleanup,
    )