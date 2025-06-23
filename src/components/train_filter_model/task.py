# src/components/train_filter_model/task.py
"""
Tarea del componente de Entrenamiento de Modelo Filtro Supervisado. (Versión con Logging Robusto)

Responsabilidades:
1.  Cargar un modelo LSTM entrenado y sus artefactos.
2.  Cargar los datos de entrenamiento.
3.  Generar predicciones con el modelo LSTM sobre los datos.
4.  Usar estas predicciones como features para entrenar un clasificador LightGBM.
5.  Guardar el modelo LightGBM entrenado en un directorio GCS versionado y limpiar versiones antiguas.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Módulos internos
from src.shared import constants, gcs_utils, indicators

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --- Reproducibilidad ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# --- Funciones Auxiliares (Lógica Original Intacta) ---
def make_model(inp_shape, lr, dr, filt, units, heads):
    x = inp = tf.keras.layers.Input(shape=inp_shape, dtype=tf.float32)
    x = tf.keras.layers.Conv1D(filt, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dr)(x)
    out = tf.keras.layers.Dense(2, dtype="float32")(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(tf.keras.optimizers.Adam(lr), loss="mae")
    return model

def to_sequences(mat, win):
    X = []
    for i in range(win, len(mat)):
        X.append(mat[i - win : i])
    return np.asarray(X, np.float32)

# --- Lógica Principal de la Tarea ---
def run_filter_training(
    lstm_model_dir: str,
    features_path: str,
    pair: str,
    timeframe: str,
    output_gcs_base_dir: str,
    trained_filter_model_path_output: Path,
    cleanup: bool = True,
):
    """
    Orquesta el proceso completo de entrenamiento del modelo de filtro.
    """
    # [LOG] Punto de control inicial.
    logger.info(f"▶️ Iniciando train_filter_model para el par '{pair}':")
    logger.info(f"  - Directorio del modelo LSTM: {lstm_model_dir}")
    logger.info(f"  - Features de entrada: {features_path}")
    logger.info(f"  - Directorio base de salida: {output_gcs_base_dir}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # [LOG] Carga de artefactos.
            logger.info("Iniciando descarga de artefactos del modelo LSTM...")
            model_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.keras", tmp_path)
            scaler_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp_path)
            params_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp_path)
            logger.info("✅ Artefactos del modelo LSTM descargados.")

            lstm_model = tf.keras.models.load_model(model_path)
            lstm_scaler = joblib.load(scaler_path)
            with open(params_path) as f:
                p = json.load(f)
            logger.info("Modelos y parámetros cargados en memoria.")

            logger.info(f"Descargando features de entrenamiento desde {features_path}...")
            local_features_path = gcs_utils.download_gcs_file(features_path, tmp_path)
            df_full = pd.read_parquet(local_features_path)
            logger.info(f"✅ Features de entrenamiento cargadas. Shape: {df_full.shape}")
            
            logger.info("Generando features y predicciones del LSTM...")
            df_ind = indicators.build_indicators(df_full.copy(), p, atr_len=14)
            feature_cols = [c for c in df_ind.columns if "atr_" not in c and c != "timestamp"]
            X_data = df_ind[feature_cols].select_dtypes(include=np.number).astype(np.float32)
            X_scaled = lstm_scaler.transform(X_data)
            X_seq = to_sequences(X_scaled, p["win"])
            
            preds = lstm_model.predict(X_seq)
            logger.info(f"Predicciones del LSTM generadas. Shape: {preds.shape}")
            
            logger.info("Generando labels para el modelo de filtro...")
            tick = 0.01 if pair.endswith("JPY") else 0.0001
            horizon = p.get("win", 20)
            closes = df_ind.close.values[p["win"]:]
            future_prices = np.roll(closes, -horizon)
            future_prices[-horizon:] = np.nan
            price_diffs = (future_prices - closes) / tick
            labels = np.where(price_diffs > (p["take_profit"] * p["stop_loss"]), 1, 0)
            valid_mask = ~np.isnan(price_diffs)
            logger.info(f"Labels generados. Shape: {labels[valid_mask].shape}")
            
            logger.info("Iniciando entrenamiento del clasificador LightGBM...")
            X_train, X_test, y_train, y_test = train_test_split(
                preds[valid_mask], labels[valid_mask], test_size=0.2, shuffle=False
            )
            
            lgb_classifier = lgb.LGBMClassifier(random_state=SEED)
            lgb_classifier.fit(X_train, y_train)
            
            accuracy = lgb_classifier.score(X_test, y_test)
            logger.info(f"✅ Entrenamiento del filtro completado. Accuracy en set de prueba: {accuracy:.4f}")
            
            ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            versioned_output_dir = f"{output_gcs_base_dir}/{pair}/{timeframe}/{ts}"
            filter_model_gcs_path = f"{versioned_output_dir}/filter_model.pkl"
            
            logger.info(f"Guardando modelo filtro en: {filter_model_gcs_path}")
            local_model_path = tmp_path / "filter_model.pkl"
            joblib.dump(lgb_classifier, local_model_path)
            
            gcs_utils.upload_gcs_file(local_model_path, filter_model_gcs_path)
            gcs_utils.verify_gcs_file_exists(filter_model_gcs_path)
            
            trained_filter_model_path_output.parent.mkdir(parents=True, exist_ok=True)
            trained_filter_model_path_output.write_text(filter_model_gcs_path)
            logger.info("Ruta de salida escrita exitosamente para KFP.")

            if cleanup:
                base_cleanup_path = f"{output_gcs_base_dir}/{pair}/{timeframe}/"
                logger.info(f"Iniciando limpieza de versiones antiguas de modelos filtro en: {base_cleanup_path}")
                gcs_utils.keep_only_latest_version(base_cleanup_path)

    except Exception as e:
        # [LOG] Captura de error fatal.
        logger.critical(f"❌ Fallo fatal en train_filter_model para el par '{pair}'. Error: {e}", exc_info=True)
        raise
    
    logger.info(f"🏁 Componente train_filter_model para '{pair}' completado exitosamente.")

# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo de filtro supervisado.")
    parser.add_argument("--lstm-model-dir", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-gcs-base-dir", required=True)
    parser.add_argument("--cleanup", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--trained-filter-model-path-output", type=Path, required=True)
    
    args = parser.parse_args()
    
    # [LOG] Registro de los argumentos recibidos.
    logger.info("Componente 'train_filter_model' iniciado con los siguientes argumentos:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
        
    run_filter_training(
        lstm_model_dir=args.lstm_model_dir,
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
        output_gcs_base_dir=args.output_gcs_base_dir,
        trained_filter_model_path_output=args.trained_filter_model_path_output,
        cleanup=args.cleanup,
    )