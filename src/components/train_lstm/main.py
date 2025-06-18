# RUTA: src/components/train_lstm/main.py
# CÓDIGO CORREGIDO — Se adhiere estrictamente a la ruta de salida proporcionada.

from __future__ import annotations
import argparse, json, logging, os, random, sys, re, tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd, tensorflow as tf
from google.cloud import storage
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, mixed_precision

# Se importa gcs_utils para usar sus funciones estandarizadas.
from src.shared import constants, gcs_utils, indicators

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
def setup_environment() -> None:
    """Configura semillas y el entorno de GPU."""
    np.random.seed(42); tf.random.set_seed(42); random.seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        # En un entorno de Vertex AI Custom Job, esto debería fallar si se pidió GPU.
        logger.warning("No se detectó GPU; continuando con CPU.")
    else:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        mixed_precision.set_global_policy("mixed_float16")
        logger.info("🚀 GPUs: %s | mixed_precision: %s",
                    [g.name for g in gpus], mixed_precision.global_policy().name)

# ─────────── Helpers de features y secuencias (sin cambios) ───────────────
def to_sequences(mat, up, dn, win):
    """Convierte una matriz de features en secuencias para el LSTM."""
    X, y_up, y_dn = [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i-win:i]); y_up.append(up[i]); y_dn.append(dn[i])
    return np.asarray(X, np.float32), np.asarray(y_up, np.float32), np.asarray(y_dn, np.float32)

def make_model(inp_sh, lr, dr, filt, units, heads):
    """Construye y compila el modelo Keras."""
    inp = layers.Input(shape=inp_sh, dtype=tf.float32)
    x   = layers.Conv1D(filt, 3, padding="same", activation="relu")(inp)
    x   = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x   = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    mdl = models.Model(inp, out)
    mdl.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mae")
    return mdl

# ───────────────────── Lógica de entrenamiento principal ────────────
# AJUSTE: La función ahora recibe la ruta de salida final y exacta.
def train_final_model(
    pair: str,
    timeframe: str,
    params_path: str,
    features_gcs_path: str,
    output_gcs_final_dir: str # <- Argumento corregido
):
    """
    Ejecuta el pipeline de entrenamiento completo y guarda los artefactos en la
    ruta GCS final proporcionada.
    """
    setup_environment()

    # Descarga de artefactos necesarios a un directorio temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        loc_params = gcs_utils.download_gcs_file(params_path, tmp_path)
        hp = json.loads(loc_params.read_text())
        
        loc_feat = gcs_utils.download_gcs_file(features_gcs_path, tmp_path)
        df_raw = pd.read_parquet(loc_feat).reset_index(drop=True)
        logger.info("Features parquet cargado — shape %s", df_raw.shape)

        # Preparación de datos y features
        df_b = indicators.build_indicators(df_raw, hp, atr_len=14)

        tick = 0.01 if pair.endswith("JPY") else 0.0001
        atr = df_b["atr_14"].values / tick
        
        # Calcular targets
        horizon = hp.get("horizon", 20)
        fut = np.roll(df_b.close.values, -horizon); fut[-horizon:] = np.nan
        diff = (fut - df_b.close.values) / tick
        up, dn = np.maximum(diff, 0)/atr, np.maximum(-diff, 0)/atr
        mask = (~np.isnan(diff))

        # Escalar features
        feat_cols = [c for c in df_b.columns if "atr_" not in c and c != "timestamp"]
        X_raw = df_b.loc[mask, feat_cols].select_dtypes(include=np.number).astype(np.float32)
        scaler = RobustScaler(); X_scaled = scaler.fit_transform(X_raw)

        # Crear secuencias y entrenar
        X_seq, y_up, y_dn = to_sequences(X_scaled, up[mask], dn[mask], hp["win"])
        logger.info("Dataset final para entrenamiento: X=%s", X_seq.shape)

        model = make_model(X_seq.shape[1:], hp["lr"], hp["dr"], hp["filt"], hp["units"], hp["heads"])
        model.fit(X_seq, np.vstack([y_up, y_dn]).T, epochs=60, batch_size=128,
                  callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=1)

        # Guardar artefactos localmente antes de subirlos.
        loc_artifacts_dir = tmp_path / "artifacts"
        loc_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        model.save(loc_artifacts_dir / "model.keras")
        joblib.dump(scaler, loc_artifacts_dir / "scaler.pkl")
        (loc_artifacts_dir / "params.json").write_text(json.dumps(hp, indent=4))
        
        # Subir el directorio de artefactos a la RUTA FINAL EXACTA proporcionada.
        gcs_utils.upload_local_directory_to_gcs(loc_artifacts_dir, output_gcs_final_dir)
        logger.info("✅ Artefactos subidos a %s", output_gcs_final_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Script de entrenamiento de modelo LSTM final.")
    ap.add_argument("--pair", required=True, help="Par de divisas, ej: EURUSD.")
    ap.add_argument("--timeframe", required=True, help="Timeframe de los datos, ej: 15minute.")
    ap.add_argument(
        "--params-file",
        dest="params_file",
        required=True,
        help="Ruta GCS al archivo JSON de parámetros optimizados.",
    )
    ap.add_argument("--features-gcs-path", required=True, help="Ruta GCS al archivo parquet de features.")
    
    # AJUSTE: El argumento de salida ahora es la ruta final y exacta.
    ap.add_argument(
        "--output-gcs-final-dir",
        required=True,
        help="Ruta GCS final y exacta donde se guardarán los artefactos del modelo."
    )
    
    a = ap.parse_args()
    
    train_final_model(
        pair=a.pair,
        timeframe=a.timeframe,
        params_path=a.params_file,
        features_gcs_path=a.features_gcs_path,
        output_gcs_final_dir=a.output_gcs_final_dir
    )