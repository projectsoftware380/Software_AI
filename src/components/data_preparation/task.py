# ---------------------------------------------------------------------
# RUTA: src/components/data_preparation/task.py
# Revisión 2025-06-12  – ajustes de robustez y limpieza
# ---------------------------------------------------------------------
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
import numpy as np                  # ← reproducibilidad
import pandas as pd

from src.shared import constants, gcs_utils, indicators

# ───────────────────── configuración global ──────────────────────────
SEED = 42                           # AJUSTE CLAVE
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ───────────────────── helpers de housekeeping ───────────────────────
def _keep_only_latest_version(base_gcs_prefix: str) -> None:
    """
    Mantiene sólo el sub-directorio con timestamp (YYYYMMDDHHMMSS)
    más reciente y borra el resto.
    """
    try:
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)

        if not base_gcs_prefix.endswith("/"):
            base_gcs_prefix += "/"

        ts_re = re.compile(r"/(\d{14})/?$")
        dirs = [p for p in fs.ls(base_gcs_prefix)
                if fs.isdir(p) and ts_re.search(p)]

        if len(dirs) <= 1:
            logger.info("No hay versiones antiguas que limpiar.")
            return

        dirs.sort(key=lambda p: ts_re.search(p).group(1), reverse=True)
        logger.info("Se conserva versión más reciente: gs://%s", dirs[0])

        for old in dirs[1:]:
            logger.info("🗑️  Eliminando versión antigua: gs://%s", old)
            fs.rm(old, recursive=True)

    except Exception as exc:
        logger.warning("No se pudo limpiar versiones antiguas: %s", exc)

# ───────────────────── validaciones de DataFrame ─────────────────────
def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas clave y elimina filas problemáticas."""
    required_cols = {"open", "high", "low", "close", "timestamp"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    # timestamp en datetime y sin NaT  – AJUSTE CLAVE
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    # OHLC deben ser numéricos y no nulos
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# ────────────────────────── tarea principal ──────────────────────────
def run_preparation(
    *,
    input_gcs_path: str,
    output_gcs_path: str,
    years_to_keep: int,
    cleanup_old_versions: bool = True,
) -> None:
    """
    1. Descarga Parquet OHLC,
    2. Calcula indicadores técnicos,
    3. Recorta ventana temporal,
    4. Sube nuevo Parquet versionado a GCS.
    """
    try:
        # 1 ▸ Cargar datos ----------------------------------------------------
        logger.info("📥 Descargando datos de: %s", input_gcs_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_in = gcs_utils.download_gcs_file(input_gcs_path, Path(tmpdir))
            df_raw = pd.read_parquet(local_in)

        df_raw = _validate_dataframe(df_raw)

        # 2 ▸ Indicadores -----------------------------------------------------
        dummy_hp: Dict[str, int] = {
            "sma_len": 50, "rsi_len": 14,
            "macd_fast": 12, "macd_slow": 26,
            "stoch_len": 14,
        }
        df_ind = indicators.build_indicators(df_raw, dummy_hp, drop_na=True)

        if df_ind.empty:
            raise RuntimeError("El DataFrame quedó vacío tras calcular indicadores.")

        # 3 ▸ Recorte ventana -------------------------------------------------
        end_ts   = df_ind["timestamp"].max()
        start_ts = end_ts - pd.DateOffset(years=years_to_keep)
        df_win   = df_ind[df_ind["timestamp"] >= start_ts].copy()

        if df_win.empty:
            raise RuntimeError(f"La ventana de {years_to_keep} años resultó vacía.")

        logger.info("🗂️  Ventana %s → %s con %s filas",
                    start_ts.date(), end_ts.date(), f"{len(df_win):,}")

        # 4 ▸ Guardar / subir Parquet ----------------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            local_out = Path(tmpdir) / "prepared_data.parquet"
            df_win.to_parquet(local_out, index=False, engine="pyarrow")
            gcs_utils.upload_gcs_file(local_out, output_gcs_path)

        logger.info("✅ Datos preparados subidos a: %s", output_gcs_path)

        # 5 ▸ Limpieza de versiones antiguas (opcional) ----------------------
        if cleanup_old_versions and output_gcs_path.startswith("gs://"):
            base_dir = "/".join(output_gcs_path.split("/")[:-2]) + "/"
            _keep_only_latest_version(base_dir)

    except Exception as exc:
        logger.critical("❌ Fallo crítico en preparación de datos: %s", exc, exc_info=True)
        raise

# ─────────────────────────── CLI / Entrypoint ─────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Task de Preparación de Datos para HPO")
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--years-to-keep", type=int, default=5)
    p.add_argument("--cleanup", type=lambda x: str(x).lower() == "true", default=True)

    # Artefacto KFP: dónde escribir la ruta final
    p.add_argument("--prepared-data-path-output", type=Path, required=True)

    args = p.parse_args()

    input_path = (
        f"{constants.DATA_PATH}/{args.pair}/{args.timeframe}/"
        f"{args.pair}_{args.timeframe}.parquet"
    )
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_path = (
        f"{constants.DATA_FILTERED_FOR_OPT_PATH}/{args.pair}/{args.timeframe}/"
        f"{ts}/{args.pair}_{args.timeframe}_recent.parquet"
    )

    run_preparation(
        input_gcs_path=input_path,
        output_gcs_path=output_path,
        years_to_keep=args.years_to_keep,
        cleanup_old_versions=args.cleanup,
    )

    logger.info("✍️  Escribiendo ruta de salida en %s", args.prepared_data_path_output)
    args.prepared_data_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.prepared_data_path_output.write_text(output_path)
