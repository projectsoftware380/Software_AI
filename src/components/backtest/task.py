# src/components/backtest/task.py
"""
Back-test completo y comparativo de la estrategia.

Genera:
- Un informe HTML detallado con QuantStats.
- CSV de operaciones para la estrategia base y la filtrada.
- Un JSON con métricas comparativas.
- Un artefacto de métricas para la UI de Vertex AI.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import quantstats as qs
import tensorflow as tf
from tensorflow.keras import models # <--- CORRECCIÓN: 'models' AÑADIDO A LA IMPORTACIÓN

from src.shared import constants, gcs_utils, indicators

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
qs.extend_pandas()

# --- Helpers ---

def _load_artifacts(
    lstm_model_dir: str, filter_model_path: str
) -> Tuple[tf.keras.Model, object, dict, lgb.LGBMClassifier, dict]:
    """Carga todos los artefactos necesarios: modelo LSTM, scaler, y el nuevo filtro LGBM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Cargar artefactos LSTM
        model = tf.keras.models.load_model(gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.keras", tmp), compile=False)
        scaler = joblib.load(gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp))
        hp = json.loads(gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp).read_text())

        # Cargar artefactos del filtro
        filter_model = joblib.load(gcs_utils.download_gcs_file(f"{filter_model_path}/filter_model.pkl", tmp))
        filter_params = json.loads(gcs_utils.download_gcs_file(f"{filter_model_path}/filter_params.json", tmp).read_text())

    logger.info("✔ Artefactos LSTM y Filtro LightGBM cargados.")
    return model, scaler, hp, filter_model, filter_params

def _prepare_backtest_data(features_path: str, hp: dict) -> pd.DataFrame:
    """Prepara los datos del hold-out para el backtest."""
    local = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    df_raw = pd.read_parquet(local)
    if df_raw.empty:
        raise ValueError("El archivo Parquet de hold-out está vacío.")

    df_ind = indicators.build_indicators(df_raw, hp, atr_len=14, drop_na=True)
    if df_ind.isna().any().any():
        raise ValueError("Persisten NaNs tras calcular indicadores en los datos de hold-out.")

    logger.info("✔ Datos de back-test (hold-out) preparados: %s filas.", len(df_ind))
    return df_ind.reset_index(drop=True)

def _generate_features_for_filter(df: pd.DataFrame, lstm_model: tf.keras.Model, scaler, hp: dict, pair: str):
    """Genera el mismo conjunto de features que se usó para entrenar el filtro."""
    feature_cols = list(scaler.feature_names_in_)
    X_scaled = scaler.transform(df[feature_cols].values)
    X_seq = np.stack([X_scaled[i - hp["win"]: i] for i in range(hp["win"], len(X_scaled))]).astype(np.float32)

    emb_model = models.Model(lstm_model.input, lstm_model.layers[-2].output)
    preds = lstm_model.predict(X_seq, verbose=0, batch_size=1024).astype(np.float32)
    embs = emb_model.predict(X_seq, verbose=0, batch_size=1024).astype(np.float32)

    df_aligned = df.iloc[hp["win"]:].copy().reset_index()
    features_df = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])])
    features_df['pred_up'] = preds[:, 0]
    features_df['pred_down'] = preds[:, 1]
    features_df['pred_diff'] = preds[:, 0] - preds[:, 1]
    
    tick = 0.01 if pair.endswith("JPY") else 0.0001
    atr = df_aligned["atr_14"].values
    
    features_df['sl_pips'] = np.where(features_df['pred_up'] > features_df['pred_down'], hp['min_thr_up'] * atr / tick, hp['min_thr_dn'] * atr / tick)
    features_df['tp_pips'] = features_df['sl_pips'] * hp['rr']
    features_df['rr_ratio'] = hp['rr']
    
    df_aligned['timestamp'] = pd.to_datetime(df_aligned['timestamp'], unit='ms')
    features_df['hour'] = df_aligned['timestamp'].dt.hour
    features_df['day_of_week'] = df_aligned['timestamp'].dt.dayofweek
    
    # Devolver también las predicciones UP/DOWN crudas para la simulación
    return features_df, preds[:, 0], preds[:, 1]

# --- NUEVA FUNCIÓN DE SIMULACIÓN AÑADIDA ---
def _run_backtest_simulation_from_signals(
    df_aligned: pd.DataFrame,
    up_preds: np.ndarray,
    dn_preds: np.ndarray,
    hp: dict,
    pair: str,
    accept_mask: np.ndarray
) -> pd.DataFrame:
    """Simula operaciones basándose en las señales del LSTM y una máscara de aceptación."""
    trades = []
    dq = deque(maxlen=hp["smooth_win"])
    tick = 0.01 if pair.endswith("JPY") else 0.0001
    horizon = hp['horizon']
    closes = df_aligned.close.values

    for i in range(len(up_preds) - horizon):
        u, d = up_preds[i], dn_preds[i]
        mag, diff = (u if u > d else d), abs(u - d)
        raw_dir = 1 if u > d else -1
        
        cond_base = (raw_dir == 1 and u >= hp["min_thr_up"]) or (raw_dir == -1 and d >= hp["min_thr_dn"]) and diff >= hp["delta_min"]
        dq.append(raw_dir if cond_base else 0)

        buys, sells = dq.count(1), dq.count(-1)
        base_signal = 1 if buys > hp["smooth_win"] // 2 else -1 if sells > hp["smooth_win"] // 2 else 0

        final_signal = base_signal if accept_mask[i] else 0

        if final_signal != 0:
            entry_time = df_aligned['timestamp'].iloc[i]
            exit_time = df_aligned['timestamp'].iloc[i + horizon]
            pnl_pips = ((closes[i + horizon] - closes[i]) / tick) * final_signal - constants.SPREADS_PIP.get(pair, 0.8)
            
            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": final_signal,
                "pnl_pips": pnl_pips
            })

    return pd.DataFrame(trades)

def _calculate_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula un conjunto extendido de métricas de rendimiento."""
    if trades_df.empty or 'pnl_pips' not in trades_df or trades_df['pnl_pips'].empty:
        return {"trades": 0, "win_rate": 0.0, "total_pips": 0.0, "sharpe": 0.0, "calmar": 0.0, "max_drawdown": 0.0}

    pnl = trades_df['pnl_pips']
    returns = pnl / 10000 
    
    metrics = {
        "trades": float(len(pnl)),
        "win_rate": float((pnl > 0).sum() / len(pnl)) if len(pnl) > 0 else 0.0,
        "total_pips": float(pnl.sum()),
        "avg_pips": float(pnl.mean()),
        "sharpe": float(qs.stats.sharpe(returns, periods=252* (24*60/15))),
        "calmar": float(qs.stats.calmar(returns)),
        "max_drawdown": float(qs.stats.max_drawdown(returns)),
    }
    return {k: (v if np.isfinite(v) else 0.0) for k, v in metrics.items()}

# --- Función Orquestadora ---

def run_backtest(*, lstm_model_dir: str, filter_model_path: str, features_path: str, pair: str, timeframe: str) -> Tuple[str, str]:
    """Orquesta el backtest completo, comparativo y genera el informe."""
    
    lstm_model, scaler, hp, filter_model, filter_params = _load_artifacts(lstm_model_dir, filter_model_path)
    df_bt = _prepare_backtest_data(features_path, hp)

    df_aligned = df_bt.iloc[hp["win"]:].copy().reset_index(drop=True)
    
    features_for_filter, up_preds, dn_preds = _generate_features_for_filter(df_bt, lstm_model, scaler, hp, pair)

    filter_threshold = filter_params['best_threshold']
    filter_probs = filter_model.predict_proba(features_for_filter)[:, 1]
    accept_mask = filter_probs >= filter_threshold

    logger.info("Ejecutando simulación para la estrategia BASE (sin filtro)...")
    base_trades = _run_backtest_simulation_from_signals(df_aligned, up_preds, dn_preds, hp, pair, np.ones_like(accept_mask, dtype=bool))
    
    logger.info("Ejecutando simulación para la estrategia FILTRADA...")
    filtered_trades = _run_backtest_simulation_from_signals(df_aligned, up_preds, dn_preds, hp, pair, accept_mask)

    metrics = {"base": _calculate_metrics(base_trades), "filtered": _calculate_metrics(filtered_trades)}
    logger.info(f"Métricas Base: {metrics['base']}")
    logger.info(f"Métricas Filtrada: {metrics['filtered']}")

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir = f"{constants.BACKTEST_RESULTS_PATH}/{pair}/{timeframe}/{ts}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        if not base_trades.empty:
            base_trades.to_csv(tmp / "trades_base.csv", index=False)
            gcs_utils.upload_gcs_file(tmp / "trades_base.csv", f"{output_dir}/trades_base.csv")

        if not filtered_trades.empty:
            df_for_report = filtered_trades.copy()
            df_for_report['returns'] = df_for_report['pnl_pips'] / 10000
            df_for_report.set_index('entry_time', inplace=True)
            
            report_path = tmp / "report_filtered.html"
            qs.reports.html(df_for_report['returns'], output=str(report_path), title=f"{pair} Filtered Strategy")
            gcs_utils.upload_gcs_file(report_path, f"{output_dir}/report_filtered.html")

            df_for_report.to_csv(tmp / "trades_filtered.csv")
            gcs_utils.upload_gcs_file(tmp / "trades_filtered.csv", f"{output_dir}/trades_filtered.csv")

        json_metrics = tmp / "metrics.json"
        json_metrics.write_text(json.dumps(metrics, indent=4))
        gcs_utils.upload_gcs_file(json_metrics, f"{output_dir}/metrics.json")
        
        kfp_json = tmp / "kfp_metrics.json"
        kfp_json.write_text(json.dumps({"metrics": [{"name": f"filtered-{k}", "numberValue": v} for k, v in metrics["filtered"].items()]}))

    logger.info("🎯 Back-test finalizado. Resultados en %s", output_dir)
    return output_dir, str(kfp_json)

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Back-test Task")
    parser.add_argument("--lstm-model-dir", required=True)
    parser.add_argument("--filter-model-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    args = parser.parse_args()

    out_dir, kfp_metrics_file = run_backtest(
        lstm_model_dir=args.lstm_model_dir,
        filter_model_path=args.filter_model_path,
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
    )
    
    Path("/tmp/backtest_dir.txt").write_text(out_dir)
    Path("/tmp/kfp_metrics.json").write_text(Path(kfp_metrics_file).read_text())