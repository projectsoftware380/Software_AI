# src/components/data_preparation/task.py
"""
Tarea del componente de preparación de datos para optimización.

Responsabilidades:
1.  Cargar el Parquet con datos OHLC e indicadores básicos.
2.  Calcular un set de indicadores técnicos robustos (usando el módulo `indicators`).
3.  Recortar el histórico a una ventana de N años recientes.
4.  Asegurar que no queden valores NaN en el dataset final.
5.  Guardar el Parquet resultante en una nueva ruta versionada en GCS.
6.  (Opcional) Limpiar versiones antiguas en el mismo directorio para ahorrar espacio.

Este script reemplaza la funcionalidad de `prepare_opt_data.py`.
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

import pandas as pd
import gcsfs

# Importar los módulos compartidos de la nueva estructura
from src.shared import constants, gcs_utils, indicators

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --- Lógica de Limpieza de Versiones Antiguas ---
def _keep_only_latest_version(base_gcs_prefix: str) -> None:
    """
    Mantiene solo el sub-directorio con el timestamp más reciente.

    Examina una ruta base en GCS (ej: .../data_filtered_for_opt_v3/<pair>/<tf>/)
    y elimina todos los subdirectorios versionados (YYYYMMDDHHMMSS) excepto
    el más nuevo.
    """
    try:
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
        # El patrón busca directorios que terminen en /YYYYMMDDHHMMSS/
        timestamp_pattern = re.compile(r"/(\d{14})/?$")
        
        all_dirs = fs.ls(base_gcs_prefix)
        versioned_dirs = [d for d in all_dirs if timestamp_pattern.search(d)]

        if len(versioned_dirs) <= 1:
            logger.info("No hay versiones antiguas que limpiar.")
            return

        # Ordenar de más nuevo a más viejo basándose en el timestamp del nombre
        dirs_sorted = sorted(
            versioned_dirs,
            key=lambda p: timestamp_pattern.search(p).group(1),
            reverse=True,
        )

        for old_dir in dirs_sorted[1:]:
            logger.info(f"🗑️  Eliminando versión anterior: gs://{old_dir}")
            fs.rm(old_dir, recursive=True)
            
    except Exception as e:
        logger.warning(f"⚠️ No se pudo realizar la limpieza de versiones antiguas en '{base_gcs_prefix}': {e}")


# --- Orquestación Principal de la Tarea ---
def run_preparation(
    input_gcs_path: str,
    output_gcs_path: str,
    years_to_keep: int,
    cleanup_old_versions: bool = True,
) -> None:
    """
    Orquesta el proceso completo de preparación de datos.
    """
    try:
        # 1. Cargar datos desde GCS
        logger.info(f"Cargando datos desde: {input_gcs_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = gcs_utils.download_gcs_file(input_gcs_path, Path(tmpdir))
            df = pd.read_parquet(local_path)
        
        if "timestamp" not in df.columns:
            raise ValueError("La columna 'timestamp' es necesaria y no se encontró.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        logger.info(f"✔ Datos cargados: {len(df):,} filas")
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        # 2. Calcular indicadores (de forma robusta)
        # Se usa un set de hiperparámetros "dummy" solo para generar los indicadores.
        # La optimización posterior elegirá los valores correctos.
        dummy_hp: Dict[str, int] = {
            "sma_len": 50, "rsi_len": 14, "macd_fast": 12,
            "macd_slow": 26, "stoch_len": 14
        }
        df_indicators = indicators.build_indicators(df, dummy_hp, drop_na=True)
        logger.info(f"🔧 Indicadores calculados -> {len(df_indicators):,} filas restantes tras limpiar NaNs.")

        # 3. Recortar a la ventana de N años
        if not df_indicators.empty:
            end_ts = df_indicators["timestamp"].max()
            start_ts = end_ts - pd.DateOffset(years=years_to_keep)
            df_window = df_indicators[df_indicators["timestamp"] >= start_ts].copy()
            
            if df_window.empty:
                raise RuntimeError(f"La ventana de {years_to_keep} años resultó vacía. Revise el rango de fechas.")
            
            logger.info(
                f"🗂️  Ventana de {years_to_keep} años ({start_ts.date()} -> {end_ts.date()}) "
                f"seleccionada -> {len(df_window):,} filas."
            )
        else:
             raise RuntimeError("El DataFrame quedó vacío después de calcular indicadores.")


        # 4. Guardar el Parquet filtrado en GCS
        with tempfile.TemporaryDirectory() as tmpdir:
            local_output_path = Path(tmpdir) / "prepared_data.parquet"
            df_window.to_parquet(local_output_path, index=False, engine="pyarrow")
            gcs_utils.upload_gcs_file(local_output_path, output_gcs_path)

        # 5. Limpiar versiones antiguas si está activado
        if cleanup_old_versions and output_gcs_path.startswith("gs://"):
            # La ruta base es el directorio padre del directorio versionado
            # ej: .../data_filtered_for_opt_v3/<pair>/<tf>/
            base_cleanup_path = "/".join(output_gcs_path.split("/")[:-2]) + "/"
            _keep_only_latest_version(base_cleanup_path)

    except Exception as e:
        logger.critical(f"❌ Fallo crítico en la preparación de datos: {e}", exc_info=True)
        raise


# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Preparación de Datos para KFP.")

    # Argumentos que el componente KFP le pasará
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--years-to-keep", type=int, default=5)
    parser.add_argument("--cleanup", type=bool, default=True)

    args = parser.parse_args()
    
    # Construir las rutas de entrada y salida usando las constantes
    input_path = f"{constants.DATA_PATH}/{args.pair}/{args.timeframe}/{args.pair}_{args.timeframe}.parquet"
    
    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_path = (
        f"{constants.DATA_FILTERED_FOR_OPT_PATH}/{args.pair}/{args.timeframe}/"
        f"{timestamp_str}/{args.pair}_{args.timeframe}_recent.parquet"
    )

    # Ejecutar la lógica principal
    run_preparation(
        input_gcs_path=input_path,
        output_gcs_path=output_path,
        years_to_keep=args.years_to_keep,
        cleanup_old_versions=args.cleanup,
    )

    # KFP necesita una forma de pasar la ruta de salida al siguiente componente
    # Imprimir la ruta de salida es una forma simple de hacerlo.
    # El `component.yaml` capturará esta salida estándar.
    print(output_path)