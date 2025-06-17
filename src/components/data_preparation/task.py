# src/components/data_preparation/task.py
"""
Tarea del componente de preparación de datos.

Responsabilidades:
1. Descargar los datos crudos OHLC para uno o todos los pares de divisas.
2. Calcular un conjunto base de indicadores técnicos.
3. Dividir los datos recientes en dos conjuntos:
   a. Un conjunto de entrenamiento/optimización (ej. 5 años menos 3 meses).
   b. Un conjunto de hold-out (ej. los últimos 3 meses) para la validación final.
4. Subir ambos archivos Parquet a GCS en carpetas versionadas por timestamp.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

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
    pair: str,
    timeframe: str,
    years_to_keep: int,
    holdout_months: int,
    cleanup_old_versions: bool,
    prepared_data_path_output: Path,
    holdout_data_path_output: Path,
) -> None:
    """
    Orquesta la preparación y división de los datos.
    """
    try:
        # AJUSTE: Si pair es "ALL", procesa todos los pares de constants.SPREADS_PIP.
        # De lo contrario, procesa solo el par especificado.
        pairs_to_process = list(constants.SPREADS_PIP.keys()) if pair == "ALL" else [pair]
        logger.info(f"Iniciando preparación de datos para los pares: {pairs_to_process}")

        # 1. Cargar y validar datos
        all_dfs = []
        for single_pair in pairs_to_process:
            pair_input_path = f"{constants.DATA_PATH}/{single_pair}/{timeframe}/{single_pair}_{timeframe}.parquet"
            logger.info("📥 Descargando datos crudos para %s de: %s", single_pair, pair_input_path)
            try:
                local_in = gcs_utils.ensure_gcs_path_and_get_local(pair_input_path)
                df_pair = pd.read_parquet(local_in)
                all_dfs.append(df_pair)
            except Exception as e:
                logger.warning(f"⚠️ No se pudieron cargar datos para {single_pair} desde {pair_input_path}: {e}")
                continue

        if not all_dfs:
            raise RuntimeError("No se descargó ningún dato para los pares especificados.")

        df_raw = pd.concat(all_dfs, ignore_index=True)
        df_raw = _validate_dataframe(df_raw)
        logger.info("✔ Datos combinados y validados. Total de filas: %s", len(df_raw))

        # 2. Calcular indicadores
        # AJUSTE: Usa los parámetros dummy de indicadores desde las constantes.
        df_ind = indicators.build_indicators(df_raw, constants.DUMMY_INDICATOR_PARAMS, drop_na=True)
        if df_ind.empty:
            raise RuntimeError("El DataFrame quedó vacío tras calcular indicadores.")

        # 3. Recorte y división de ventanas
        end_ts = df_ind["timestamp"].max()
        start_ts = end_ts - pd.DateOffset(years=years_to_keep)
        holdout_split_ts = end_ts - pd.DateOffset(months=holdout_months)

        df_full_window = df_ind[df_ind["timestamp"] >= start_ts].copy()
        if df_full_window.empty:
            raise RuntimeError(f"La ventana de {years_to_keep} años resultó vacía.")

        df_train_opt = df_full_window[df_full_window["timestamp"] < holdout_split_ts]
        df_holdout = df_full_window[df_full_window["timestamp"] >= holdout_split_ts]

        if df_train_opt.empty or df_holdout.empty:
            raise RuntimeError("La división entre entrenamiento y hold-out resultó en un conjunto vacío.")

        logger.info(f"🗂️  Datos de Entrenamiento/Optimización: {len(df_train_opt):,} filas")
        logger.info(f"🔒 Datos de Hold-out: {len(df_holdout):,} filas")

        # AJUSTE: El componente ahora construye sus propias rutas de salida versionadas.
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        output_base_dir = f"{constants.DATA_FILTERED_FOR_OPT_PATH}/{pair}/{timeframe}/{ts}"
        output_gcs_path_main = f"{output_base_dir}/{pair}_{timeframe}_train_opt.parquet"
        output_gcs_path_holdout = f"{output_base_dir}/{pair}_{timeframe}_holdout.parquet"

        # 4. Guardar y subir ambos Parquets
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_main = tmp_path / "prepared_data.parquet"
            df_train_opt.to_parquet(local_main, index=False, engine="pyarrow")
            gcs_utils.upload_gcs_file(local_main, output_gcs_path_main)
            logger.info("✅ Datos de entrenamiento subidos a: %s", output_gcs_path_main)

            local_holdout = tmp_path / "holdout_data.parquet"
            df_holdout.to_parquet(local_holdout, index=False, engine="pyarrow")
            gcs_utils.upload_gcs_file(local_holdout, output_gcs_path_holdout)
            logger.info("✅ Datos de hold-out subidos a: %s", output_gcs_path_holdout)

        # 5. Limpieza de versiones antiguas (si está activado)
        if cleanup_old_versions:
            # === INICIO DE LA CORRECCIÓN ===
            # Se obtiene el directorio padre manipulando la cadena de texto,
            # lo que es seguro para rutas GCS.
            parent_dir = '/'.join(output_base_dir.split('/')[:-1])
            # === FIN DE LA CORRECCIÓN ===
            gcs_utils.keep_only_latest_version(parent_dir)

        # AJUSTE: Escribir las rutas de salida a los archivos que KFP espera.
        prepared_data_path_output.parent.mkdir(parents=True, exist_ok=True)
        prepared_data_path_output.write_text(output_gcs_path_main)
        
        holdout_data_path_output.parent.mkdir(parents=True, exist_ok=True)
        holdout_data_path_output.write_text(output_gcs_path_holdout)

        logger.info("✍️  Rutas de salida para KFP escritas con éxito.")

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

    # AJUSTE: Argumentos de salida para KFP. Ya no se construyen rutas aquí.
    p.add_argument("--prepared-data-path-output", type=Path, required=True)
    p.add_argument("--holdout-data-path-output", type=Path, required=True)

    args = p.parse_args()

    run_preparation(
        pair=args.pair,
        timeframe=args.timeframe,
        years_to_keep=args.years_to_keep,
        holdout_months=args.holdout_months,
        cleanup_old_versions=args.cleanup,
        prepared_data_path_output=args.prepared_data_path_output,
        holdout_data_path_output=args.holdout_data_path_output,
    )