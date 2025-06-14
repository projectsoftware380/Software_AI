# src/components/data_preparation/task.py
"""
Tarea del componente de preparación de datos.

Responsabilidades:
1. Descargar los datos crudos OHLC de un par/timeframe.
2. Calcular un conjunto base de indicadores técnicos.
3. Dividir los datos recientes en dos conjuntos:
   a. Un conjunto de entrenamiento/optimización (ej. 5 años menos 3 meses).
   b. Un conjunto de hold-out (ej. los últimos 3 meses) para la validación final.
4. Subir ambos archivos Parquet a GCS en carpetas versionadas.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict

import gcsfs
import numpy as np
import pandas as pd

from src.shared import constants, gcs_utils, indicators

# --- Configuración Global ---
SEED = 42
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- Helpers ---
def _keep_only_latest_version(base_gcs_prefix: str) -> None:
    """Mantiene sólo el sub-directorio con timestamp más reciente y borra el resto."""
    try:
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
        if not base_gcs_prefix.endswith("/"):
            base_gcs_prefix += "/"

        ts_re = re.compile(r"/(\d{14})/?$")
        dirs = [p for p in fs.ls(base_gcs_prefix) if fs.isdir(p) and ts_re.search(p)]

        if len(dirs) <= 1:
            return

        dirs.sort(key=lambda p: ts_re.search(p).group(1), reverse=True)
        for old in dirs[1:]:
            logger.info("🗑️  Eliminando versión antigua de datos preparados: gs://%s", old)
            fs.rm(old, recursive=True)
    except Exception as exc:
        logger.warning("No se pudo limpiar versiones antiguas: %s", exc)

def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas clave y elimina filas problemáticas."""
    required_cols = {"open", "high", "low", "close", "timestamp"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Faltan columnas obligatorias: {required_cols - set(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# --- Tarea Principal ---
def run_preparation(
    *,
    input_gcs_path: str,
    output_gcs_path_main: str,
    output_gcs_path_holdout: str,
    years_to_keep: int,
    holdout_months: int,
    cleanup_old_versions: bool = True,
) -> None:
    """
    Orquesta la preparación y división de los datos.
    """
    try:
        # 1. Cargar y validar datos
        logger.info("📥 Descargando datos crudos de: %s", input_gcs_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_in = gcs_utils.download_gcs_file(input_gcs_path, Path(tmpdir))
            df_raw = pd.read_parquet(local_in)
        df_raw = _validate_dataframe(df_raw)

        # 2. Calcular indicadores
        dummy_hp: Dict[str, int] = {
            "sma_len": 50, "rsi_len": 14,
            "macd_fast": 12, "macd_slow": 26,
            "stoch_len": 14,
        }
        df_ind = indicators.build_indicators(df_raw, dummy_hp, drop_na=True)
        if df_ind.empty:
            raise RuntimeError("El DataFrame quedó vacío tras calcular indicadores.")

        # 3. Recorte y división de ventanas
        end_ts = df_ind["timestamp"].max()
        start_ts = end_ts - pd.DateOffset(years=years_to_keep)
        holdout_split_ts = end_ts - pd.DateOffset(months=holdout_months)
        
        df_full_window = df_ind[df_ind["timestamp"] >= start_ts].copy()
        if df_full_window.empty:
            raise RuntimeError(f"La ventana de {years_to_keep} años resultó vacía.")

        # Dividir en entrenamiento/optimización y hold-out
        df_train_opt = df_full_window[df_full_window["timestamp"] < holdout_split_ts]
        df_holdout = df_full_window[df_full_window["timestamp"] >= holdout_split_ts]

        if df_train_opt.empty or df_holdout.empty:
            raise RuntimeError("La división entre entrenamiento y hold-out resultó en un conjunto vacío.")

        logger.info(f"🗂️  Datos de Entrenamiento/Optimización: {len(df_train_opt):,} filas")
        logger.info(f"🔒 Datos de Hold-out: {len(df_holdout):,} filas")

        # 4. Guardar y subir ambos Parquets
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Guardar y subir el archivo principal
            local_main = tmp_path / "prepared_data.parquet"
            df_train_opt.to_parquet(local_main, index=False, engine="pyarrow")
            gcs_utils.upload_gcs_file(local_main, output_gcs_path_main)
            logger.info("✅ Datos de entrenamiento subidos a: %s", output_gcs_path_main)

            # Guardar y subir el archivo hold-out
            local_holdout = tmp_path / "holdout_data.parquet"
            df_holdout.to_parquet(local_holdout, index=False, engine="pyarrow")
            gcs_utils.upload_gcs_file(local_holdout, output_gcs_path_holdout)
            logger.info("✅ Datos de hold-out subidos a: %s", output_gcs_path_holdout)

        # 5. Limpieza de versiones antiguas (si está activado)
        if cleanup_old_versions and output_gcs_path_main.startswith("gs://"):
            base_dir = "/".join(output_gcs_path_main.split("/")[:-2]) + "/"
            _keep_only_latest_version(base_dir)

    except Exception as exc:
        logger.critical("❌ Fallo crítico en preparación de datos: %s", exc, exc_info=True)
        raise

# --- Punto de Entrada / CLI ---
if __name__ == "__main__":
    p = argparse.ArgumentParser("Task de Preparación y División de Datos")
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--years-to-keep", type=int, default=5)
    p.add_argument("--holdout-months", type=int, default=3)
    p.add_argument("--cleanup", type=lambda x: str(x).lower() == "true", default=True)

    # Definir las dos rutas de salida para KFP
    p.add_argument("--prepared-data-path-output", type=Path, required=True)
    p.add_argument("--holdout-data-path-output", type=Path, required=True)

    args = p.parse_args()

    # Construir rutas de entrada y salida
    input_path = f"{constants.DATA_PATH}/{args.pair}/{args.timeframe}/{args.pair}_{args.timeframe}.parquet"
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    
    # Ruta para el archivo principal (entrenamiento/optimización)
    output_path_main = (
        f"{constants.DATA_FILTERED_FOR_OPT_PATH}/{args.pair}/{args.timeframe}/"
        f"{ts}/{args.pair}_{args.timeframe}_train_opt.parquet"
    )
    # Ruta para el archivo hold-out (backtesting final)
    output_path_holdout = (
        f"{constants.DATA_FILTERED_FOR_OPT_PATH}/{args.pair}/{args.timeframe}/"
        f"{ts}/{args.pair}_{args.timeframe}_holdout.parquet"
    )

    run_preparation(
        input_gcs_path=input_path,
        output_gcs_path_main=output_path_main,
        output_gcs_path_holdout=output_path_holdout,
        years_to_keep=args.years_to_keep,
        holdout_months=args.holdout_months,
        cleanup_old_versions=args.cleanup,
    )

    # Escribir ambas rutas de salida para que KFP las pueda usar
    args.prepared_data_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.prepared_data_path_output.write_text(output_path_main)
    
    args.holdout_data_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.holdout_data_path_output.write_text(output_path_holdout)

    logger.info("✍️  Rutas de salida para datos de entrenamiento y hold-out escritas con éxito.")