# src/components/train_filter_model/task.py
"""
Entrena un clasificador LightGBM para filtrar las señales del LSTM.

Flujo de Trabajo:
1.  Carga el modelo LSTM entrenado y los datos preparados desde rutas exactas.
2.  Genera "features" y "labels" para el filtro.
3.  Entrena un modelo LightGBM para predecir operaciones exitosas.
4.  Encuentra un umbral de probabilidad óptimo en un set de validación.
5.  Guarda el modelo y sus parámetros en una nueva ruta GCS versionada.
6.  Propaga la ruta exacta del directorio del modelo al siguiente componente.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import models

from src.shared import constants, gcs_utils, indicators

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Helpers ---
# La lógica de _generate_features_and_labels no necesita cambios funcionales.
def _generate_features_and_labels(df_raw: pd.DataFrame, lstm_model: tf.keras.Model, scaler, hp: dict, pair: str):
    """Genera el dataset completo (features y labels) para el clasificador."""
    
    # 1. Generar predicciones y embeddings del LSTM
    df_ind = indicators.build_indicators(df_raw, hp, atr_len=14)
    feature_cols = list(scaler.feature_names_in_)
    X_scaled = scaler.transform(df_ind[feature_cols])
    X_seq = np.stack([X_scaled[i - hp["win"]: i] for i in range(hp["win"], len(X_scaled))]).astype(np.float32)
    
    emb_model = models.Model(lstm_model.input, lstm_model.layers[-2].output)
    preds = lstm_model.predict(X_seq, verbose=0).astype(np.float32)
    embs = emb_model.predict(X_seq, verbose=0).astype(np.float32)
    
    # 2. Crear features adicionales
    df_aligned = df_ind.iloc[hp["win"]:].copy().reset_index()
    features_df = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])])
    features_df['pred_up'] = preds[:, 0]
    features_df['pred_down'] = preds[:, 1]
    features_df['pred_diff'] = preds[:, 0] - preds[:, 1]
    
    tick = 0.01 if pair.endswith("JPY") else 0.0001
    atr = df_aligned[f"atr_14"].values
    
    features_df['sl_pips'] = np.where(features_df['pred_up'] > features_df['pred_down'],
                                     hp['min_thr_up'] * atr / tick,
                                     hp['min_thr_dn'] * atr / tick)
    features_df['tp_pips'] = features_df['sl_pips'] * hp['rr']
    features_df['rr_ratio'] = hp['rr']
    
    df_aligned['timestamp'] = pd.to_datetime(df_aligned['timestamp'], unit='ms')
    features_df['hour'] = df_aligned['timestamp'].dt.hour
    features_df['day_of_week'] = df_aligned['timestamp'].dt.dayofweek

    # 3. Generar etiquetas (1 si TP, 0 si SL)
    labels = []
    closes = df_aligned.close.values
    horizon = hp['horizon']
    
    for i in range(len(closes) - horizon):
        entry_price = closes[i]
        sl_price = entry_price - features_df['sl_pips'].iloc[i] * tick if features_df['pred_up'].iloc[i] > features_df['pred_down'].iloc[i] else entry_price + features_df['sl_pips'].iloc[i] * tick
        tp_price = entry_price + features_df['tp_pips'].iloc[i] * tick if features_df['pred_up'].iloc[i] > features_df['pred_down'].iloc[i] else entry_price - features_df['tp_pips'].iloc[i] * tick

        future_prices = closes[i+1 : i+1+horizon]
        
        hit_tp = np.where(future_prices >= tp_price)[0] if features_df['pred_up'].iloc[i] > features_df['pred_down'].iloc[i] else np.where(future_prices <= tp_price)[0]
        hit_sl = np.where(future_prices <= sl_price)[0] if features_df['pred_up'].iloc[i] > features_df['pred_down'].iloc[i] else np.where(future_prices >= sl_price)[0]

        tp_time = hit_tp[0] if len(hit_tp) > 0 else np.inf
        sl_time = hit_sl[0] if len(hit_sl) > 0 else np.inf

        if tp_time < sl_time:
            labels.append(1) # Éxito
        elif sl_time < tp_time:
            labels.append(0) # Fracaso
        else:
            labels.append(-1) # No resuelta
    
    features_df = features_df.iloc[:len(labels)].copy()
    features_df['label'] = labels
    dataset = features_df[features_df['label'] != -1].copy()
    
    X = dataset.drop('label', axis=1)
    y = dataset['label']

    return X, y


def _make_checkpoint_callback(path: Path) -> callable:
    """Crea un callback de LightGBM que guarda el mejor modelo según val_loss."""

    best_loss = [float("inf")]

    def _callback(env: lgb.callback.CallbackEnv) -> None:
        if not env.evaluation_result_list:
            return
        # env.evaluation_result_list: [(data_name, eval_name, result, is_higher_better), ...]
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            if data_name == "valid_0" and eval_name == "binary_logloss" and result < best_loss[0]:
                best_loss[0] = result
                env.model.save_model(str(path), num_iteration=env.iteration + 1)

    _callback.order = 0
    return _callback

# --- Función Principal ---
def run_filter_training(
    *,
    lstm_model_dir: str,
    features_path: str,
    pair: str,
    timeframe: str,
    output_gcs_base_dir: str,
    trained_filter_model_path_output: Path,
) -> None:
    """Orquesta el entrenamiento del clasificador de filtro."""
    
    # 1. Cargar artefactos LSTM
    logger.info("Cargando artefactos del modelo LSTM desde %s", lstm_model_dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # AJUSTE: Se asume que el directorio lstm_model_dir ya es la ruta exacta y versionada.
        model_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.keras", tmp)
        scaler_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp)
        params_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp)
        lstm_model = models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        hp = json.loads(params_path.read_text())

        # 🔹 Asegurar que el JSON contenga los hiperparámetros de indicadores
        # necesarios para `build_indicators`.  Algunos archivos antiguos podían
        # carecer de estas claves, lo que provocaba un ``KeyError`` al intentar
        # generar los indicadores.  Para garantizar robustez, se completan con
        # los valores por defecto definidos en ``constants.DUMMY_INDICATOR_PARAMS``.
        for k, v in constants.DUMMY_INDICATOR_PARAMS.items():
            hp.setdefault(k, v)

    # 2. Generar el dataset de entrenamiento para el filtro
    local_features = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    df_raw = pd.read_parquet(local_features)
    
    logger.info("Generando features y labels para el filtro...")
    X, y = _generate_features_and_labels(df_raw, lstm_model, scaler, hp, pair)
    logger.info(f"Dataset generado con {len(X)} muestras. Distribución:\n{y.value_counts(normalize=True)}")

    # 3. Entrenar el modelo LightGBM
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    n_neg, n_pos = y_train.value_counts().sort_index()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
    )

    logger.info("Entrenando el modelo de filtro LightGBM...")
    evals_result: dict[str, list] = {}
    with tempfile.TemporaryDirectory() as fit_tmpdir:
        best_model_file = Path(fit_tmpdir) / "best_lgb.txt"
        callbacks = [
            lgb.early_stopping(10),
            lgb.record_evaluation(evals_result),
            _make_checkpoint_callback(best_model_file),
        ]
        lgb_clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )

        # Cargar explícitamente el mejor modelo antes de continuar.
        if best_model_file.exists():
            logger.info("Recuperando el modelo almacenado en %s", best_model_file)
            lgb_clf._Booster = lgb.Booster(model_file=str(best_model_file))

    # 4. Seleccionar umbral óptimo
    logger.info("Buscando umbral de probabilidad óptimo...")
    y_pred_probs = lgb_clf.predict_proba(X_val)[:, 1]
    best_expectancy, best_threshold = -np.inf, 0.5
    for threshold in np.arange(0.5, 0.9, 0.01):
        accepted_trades = X_val[(y_pred_probs >= threshold)]
        if accepted_trades.empty: continue
        pnl = np.where(y_val[y_pred_probs >= threshold] == 1, accepted_trades['tp_pips'], -accepted_trades['sl_pips'])
        if pnl.mean() > best_expectancy:
            best_expectancy = pnl.mean()
            best_threshold = threshold
    logger.info(f"Mejor umbral: {best_threshold:.2f} (Expectativa: {best_expectancy:.2f} pips)")

    # AJUSTE: Crear una ruta de salida única y versionada para este entrenamiento.
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_gcs_dir = f"{output_gcs_base_dir}/{pair}/{timeframe}/{ts}"
    
    filter_params = {"best_threshold": best_threshold, "model_type": "lightgbm"}
    
    # 5. Guardar artefactos en la ruta versionada
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        joblib.dump(lgb_clf, tmp_path / "filter_model.pkl")
        (tmp_path / "filter_params.json").write_text(json.dumps(filter_params, indent=2))
        
        gcs_utils.upload_gcs_file(tmp_path / "filter_model.pkl", f"{output_gcs_dir}/filter_model.pkl")
        gcs_utils.upload_gcs_file(tmp_path / "filter_params.json", f"{output_gcs_dir}/filter_params.json")

    logger.info("✅ Modelo de filtro y parámetros guardados en %s", output_gcs_dir)
    
    # 6. Limpiar versiones antiguas
    gcs_utils.keep_only_latest_version(f"{output_gcs_base_dir}/{pair}/{timeframe}")
    
    # AJUSTE CRÍTICO: Propagar la ruta del DIRECTORIO versionado al siguiente componente.
    trained_filter_model_path_output.parent.mkdir(parents=True, exist_ok=True)
    trained_filter_model_path_output.write_text(output_gcs_dir)
    logger.info("✍️  Ruta de salida del directorio del filtro (%s) escrita para KFP.", output_gcs_dir)

# --- Punto de Entrada ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Filter Model Task")
    parser.add_argument("--lstm-model-dir", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-gcs-base-dir", required=True)
    parser.add_argument("--trained-filter-model-path-output", type=Path, required=True)
    args = parser.parse_args()

    run_filter_training(
        lstm_model_dir=args.lstm_model_dir,
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
        output_gcs_base_dir=args.output_gcs_base_dir,
        trained_filter_model_path_output=args.trained_filter_model_path_output,
    )