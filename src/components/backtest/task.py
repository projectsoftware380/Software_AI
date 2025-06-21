# src/components/backtest/task.py
"""
Tarea del componente de Backtesting Final.

Responsabilidades:
1.  Cargar los artefactos necesarios: modelo LSTM, modelo filtro, y datos de hold-out.
2.  Generar predicciones con ambos modelos sobre los datos de hold-out.
3.  Simular las operaciones de trading basadas en las señales de los modelos.
4.  Calcular un conjunto completo de métricas de rendimiento (Sharpe, Sortino, etc.).
5.  Guardar las métricas y el registro de operaciones en un directorio GCS versionado.
6.  Limpiar las versiones antiguas de los resultados del backtest.
7.  Generar un artefacto de Métricas de KFP para visualización en la UI de Vertex.
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.shared import constants, gcs_utils, indicators

# --- Configuración (Sin Cambios) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Funciones de Métricas (Lógica Original Intacta) ---
def calculate_metrics(returns: np.ndarray, trades: pd.DataFrame) -> dict:
    if len(returns) == 0:
        return {
            "sharpe_ratio": 0, "sortino_ratio": 0, "num_trades": 0, "win_rate": 0,
            "profit_factor": 0, "max_drawdown": 0, "avg_return_per_trade": 0
        }
    
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-9
    sortino = np.mean(returns) / downside_std if downside_std > 0 else 0
    
    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / (peak + 1e-9)
    max_drawdown = np.max(drawdown)
    
    num_trades = len(trades)
    win_rate = (trades['pnl'] > 0).sum() / num_trades if num_trades > 0 else 0
    
    total_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    total_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    return {
        "sharpe_ratio": round(sharpe * np.sqrt(252), 4),
        "sortino_ratio": round(sortino * np.sqrt(252), 4),
        "num_trades": int(num_trades),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_drawdown, 4),
        "avg_return_per_trade": round(np.mean(returns) if len(returns) > 0 else 0, 4)
    }

# --- Funciones Auxiliares (Lógica Original Intacta) ---
def to_sequences(mat, win):
    X = []
    for i in range(win, len(mat)):
        X.append(mat[i - win : i])
    return np.asarray(X, np.float32)

# --- Lógica Principal de la Tarea (Ajustada) ---
def run_backtest(
    lstm_model_dir: str,
    filter_model_path: str,
    features_path: str,
    pair: str,
    timeframe: str,
    output_gcs_dir_output: Path,
    kfp_metrics_artifact_output: Path,
    cleanup: bool = True, # <-- AJUSTE: Recibe el flag de limpieza
):
    """
    Orquesta el proceso completo de backtesting para un par específico.
    """
    logger.info(f"--- Iniciando backtest final para el par: {pair} ---")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        logger.info("Cargando artefactos de modelos y datos...")
        local_lstm_model_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.keras", tmp_path)
        local_scaler_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp_path)
        local_params_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp_path)
        local_filter_model_path = gcs_utils.download_gcs_file(filter_model_path, tmp_path)
        local_features_path = gcs_utils.download_gcs_file(features_path, tmp_path)

        lstm_model = tf.keras.models.load_model(local_lstm_model_path)
        lstm_scaler = joblib.load(local_scaler_path)
        filter_model = joblib.load(local_filter_model_path)
        with open(local_params_path) as f:
            p = json.load(f)

        df_holdout = pd.read_parquet(local_features_path)
        
        df_ind = indicators.build_indicators(df_holdout.copy(), p, atr_len=14)
        feature_cols = [c for c in df_ind.columns if "atr_" not in c and c != "timestamp"]
        X_data = df_ind[feature_cols].select_dtypes(include=np.number).astype(np.float32)
        X_scaled = lstm_scaler.transform(X_data)
        
        X_seq = to_sequences(X_scaled, p["win"])
        lstm_preds = lstm_model.predict(X_seq)
        
        filter_preds = filter_model.predict_proba(lstm_preds)[:, 1]
        
        is_trade_allowed = filter_preds > 0.5
        
        up_pred, dn_pred = lstm_preds[:, 0], lstm_preds[:, 1]
        is_buy = (up_pred > p["buy_threshold"]) & is_trade_allowed
        is_sell = (dn_pred > p["sell_threshold"]) & is_trade_allowed
        
        tick = 0.01 if pair.endswith("JPY") else 0.0001
        horizon = p.get("win", 20)
        
        closes = df_ind.close.values[p["win"]:]
        
        future_prices = np.roll(closes, -horizon)
        future_prices[-horizon:] = np.nan
        
        price_diffs = (future_prices - closes) / tick
        
        y_up_val = np.maximum(price_diffs, 0)
        y_dn_val = np.maximum(-price_diffs, 0)

        returns = np.zeros(len(y_up_val))
        returns[is_buy] = y_up_val[is_buy] - (p["take_profit"] * p["stop_loss"])
        returns[is_sell] = np.where(y_dn_val[is_sell] < p["stop_loss"], p["stop_loss"] - y_dn_val[is_sell], 0)
        
        trades = pd.DataFrame({
            "timestamp": df_ind.index[p["win"]:][is_buy | is_sell],
            "pnl": returns[is_buy | is_sell]
        })
        
        metrics = calculate_metrics(returns, trades)
        logger.info(f"Métricas de backtest para {pair}: {metrics}")
        
        # --- AJUSTE: Lógica de guardado en directorio versionado ---
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        versioned_output_dir_str = f"{constants.BACKTEST_RESULTS_PATH}/{pair}/{timeframe}/{ts}"
        
        # Guardar resultados en GCS
        trades.to_csv(f"{versioned_output_dir_str}/trades.csv", index=False)
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp_metrics:
            json.dump(metrics, tmp_metrics, indent=4)
            gcs_utils.upload_gcs_file(Path(tmp_metrics.name), f"{versioned_output_dir_str}/metrics.json")
        
        # Escribir la ruta del directorio de salida para KFP
        output_gcs_dir_output.parent.mkdir(parents=True, exist_ok=True)
        output_gcs_dir_output.write_text(versioned_output_dir_str)
        
        # Generar artefacto de métricas para KFP
        kfp_metrics_artifact_output.parent.mkdir(parents=True, exist_ok=True)
        with open(kfp_metrics_artifact_output, "w") as f:
            json.dump({
                "metrics": [{"name": k.replace("_", "-"), "numberValue": v, "format": "RAW"} for k, v in metrics.items()]
            }, f)
        
        logger.info(f"✅ Backtest para {pair} completado y resultados guardados en {versioned_output_dir_str}")

        # --- AJUSTE AÑADIDO: LÓGICA DE LIMPIEZA ---
        if cleanup:
            base_cleanup_path = f"{constants.BACKTEST_RESULTS_PATH}/{pair}/{timeframe}/"
            logger.info(f"Iniciando limpieza de versiones antiguas de resultados de backtest en: {base_cleanup_path}")
            gcs_utils.keep_only_latest_version(base_cleanup_path)
        # --- FIN DEL AJUSTE ---

# --- Punto de Entrada para Ejecución como Script (Ajustado) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el backtest final.")
    
    parser.add_argument("--lstm-model-dir", required=True)
    parser.add_argument("--filter-model-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--cleanup", type=lambda x: (str(x).lower() == 'true'), default=True) # <-- AJUSTE
    parser.add_argument("--output-gcs-dir-output", type=Path, required=True)
    parser.add_argument("--kfp-metrics-artifact-output", type=Path, required=True)
    
    args = parser.parse_args()
    
    run_backtest(
        lstm_model_dir=args.lstm_model_dir,
        filter_model_path=args.filter_model_path,
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
        output_gcs_dir_output=args.output_gcs_dir_output,
        kfp_metrics_artifact_output=args.kfp_metrics_artifact_output,
        cleanup=args.cleanup, # <-- AJUSTE
    )