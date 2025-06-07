# src/components/data_preparation/task.py (Implementando tu solución)
from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import gcsfs

from src.shared import constants, gcs_utils, indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def _keep_only_latest_version(base_gcs_prefix: str) -> None:
    try:
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
        timestamp_pattern = re.compile(r"/(\d{14})/?$")
        all_dirs = fs.ls(base_gcs_prefix)
        versioned_dirs = [d for d in all_dirs if timestamp_pattern.search(d)]
        if len(versioned_dirs) <= 1:
            logger.info("No hay versiones antiguas que limpiar.")
            return
        dirs_sorted = sorted(versioned_dirs, key=lambda p: timestamp_pattern.search(p).group(1), reverse=True)
        for old_dir in dirs_sorted[1:]:
            logger.info(f"🗑️  Eliminando versión anterior: gs://{old_dir}")
            fs.rm(old_dir, recursive=True)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo realizar la limpieza de versiones antiguas en '{base_gcs_prefix}': {e}")

def run_preparation(
    input_gcs_path: str,
    output_gcs_path: str,
    years_to_keep: int,
    cleanup_old_versions: bool = True,
) -> str:
    logger.info(f"Cargando datos desde: {input_gcs_path}")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = gcs_utils.download_gcs_file(input_gcs_path, Path(tmpdir))
        df = pd.read_parquet(local_path)
    
    if "timestamp" not in df.columns:
        raise ValueError("La columna 'timestamp' es necesaria y no se encontró.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    logger.info(f"✔ Datos cargados: {len(df):,} filas")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    dummy_hp: Dict[str, int] = {
        "sma_len": 50, "rsi_len": 14, "macd_fast": 12,
        "macd_slow": 26, "stoch_len": 14
    }
    df_indicators = indicators.build_indicators(df, dummy_hp, drop_na=True)
    logger.info(f"🔧 Indicadores calculados -> {len(df_indicators):,} filas restantes tras limpiar NaNs.")

    if not df_indicators.empty:
        end_ts = df_indicators["timestamp"].max()
        start_ts = end_ts - pd.DateOffset(years=years_to_keep)
        df_window = df_indicators[df_indicators["timestamp"] >= start_ts].copy()
        if df_window.empty:
            raise RuntimeError(f"La ventana de {years_to_keep} años resultó vacía. Revise el rango de fechas.")
        logger.info(f"🗂️  Ventana de {years_to_keep} años ({start_ts.date()} -> {end_ts.date()}) seleccionada -> {len(df_window):,} filas.")
    else:
        raise RuntimeError("El DataFrame quedó vacío después de calcular indicadores.")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_output_path = Path(tmpdir) / "prepared_data.parquet"
        df_window.to_parquet(local_output_path, index=False, engine="pyarrow")
        gcs_utils.upload_gcs_file(local_output_path, output_gcs_path)

    if cleanup_old_versions and output_gcs_path.startswith("gs://"):
        base_cleanup_path = "/".join(output_gcs_path.split("/")[:-2]) + "/"
        _keep_only_latest_version(base_cleanup_path)
        
    return output_gcs_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Preparación de Datos para KFP.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--years-to-keep", type=int, default=5)
    parser.add_argument("--cleanup", type=bool, default=True)
    # Argumento para el archivo de salida que KFP proporciona
    parser.add_argument("--output-path", type=str, required=True)
    
    args = parser.parse_args()
    
    input_path = f"{constants.DATA_PATH}/{args.pair}/{args.timeframe}/{args.pair}_{args.timeframe}.parquet"
    
    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_gcs_path = (
        f"{constants.DATA_FILTERED_FOR_OPT_PATH}/{args.pair}/{args.timeframe}/"
        f"{timestamp_str}/{args.pair}_{args.timeframe}_recent.parquet"
    )

    final_gcs_path = run_preparation(
        input_gcs_path=input_path,
        output_gcs_path=output_gcs_path,
        years_to_keep=args.years_to_keep,
        cleanup_old_versions=args.cleanup,
    )

    # Escribir la ruta GCS en el archivo de salida que KFP espera (tu propuesta)
    out_path_obj = Path(args.output_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_path_obj.write_text(final_gcs_path)