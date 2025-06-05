#!/usr/bin/env python3
"""
prepare_opt_data.py
────────────────────
Genera el subconjunto de datos históricos que se usará para la búsqueda de
hiperparámetros (“optimización”) de los modelos LSTM.

• Lee un Parquet OHLC completo ―local o gs://―.
• Garantiza que las columnas esenciales se llamen: open, high, low, close,
  volume (opcional) y timestamp (datetime UTC).
• Elimina filas con NaNs en OHLC.
• Filtra los últimos N años (por defecto 5).
• Guarda el resultado en la ruta de salida (local o gs://).

Ejemplo CLI
-----------
python prepare_opt_data.py \
  --input_path  gs://bucket/data/EURUSD_15minute.parquet \
  --output_path gs://bucket/data_filtered_for_opt/EURUSD_15minute_recent.parquet \
  --years 5
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

# ─────────────── Config logging ────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────── GCS helpers ───────────────────
try:
    import gcsfs
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:  # fallback defensivo si faltan deps (no debería ocurrir en la imagen)
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "gcsfs", "google-cloud-storage"]
    )
    import gcsfs
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError


def _gcs_client() -> storage.Client:
    try:
        return storage.Client()  # usa ADC (cuenta de servicio) o cred locales
    except DefaultCredentialsError as e:
        log.critical(
            "❌ No fue posible obtener credenciales de GCP. "
            "Asegúrate de que GOOGLE_APPLICATION_CREDENTIALS esté definida "
            "o de ejecutar `gcloud auth application-default login` en local."
        )
        raise e


def _download_gs(uri: str) -> Path:
    if not uri.startswith("gs://"):
        raise ValueError("Sólo se aceptan URIs que empiecen con gs://")
    bucket_name, blob_name = uri[5:].split("/", 1)
    local = Path(tempfile.mkdtemp()) / Path(blob_name).name
    _gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(local)
    log.info(f"📥  Descargado {uri}  →  {local}")
    return local


def _upload_gs(local: Path, uri: str):
    if not uri.startswith("gs://"):
        raise ValueError("La ruta de salida debe empezar por gs://")
    bucket_name, blob_name = uri[5:].split("/", 1)
    _gcs_client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local))
    log.info(f"☁️  Subido {local}  →  {uri}")


def _read_parquet(path: str) -> pd.DataFrame:
    """Lee un Parquet desde local o GCS con pyarrow."""
    if path.startswith("gs://"):
        # gcsfs usa ‘token="cloud"’ para ADC / cred. de la VM
        return pd.read_parquet(path, engine="pyarrow", storage_options={"token": "cloud"})
    return pd.read_parquet(path, engine="pyarrow")


def _write_parquet(df: pd.DataFrame, path: str):
    """Guarda Parquet a local o GCS (mantiene compresión por defecto)."""
    if path.startswith("gs://"):
        df.to_parquet(path, index=False, engine="pyarrow", storage_options={"token": "cloud"})
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")


# ─────────────── Main logic ────────────────────
ESSENTIAL = ["open", "high", "low", "close", "timestamp"]  # volume opcional


def main():
    ap = argparse.ArgumentParser(description="Filtra últimos N años de datos OHLC.")
    ap.add_argument("--input_path", required=True, help="Parquet completo (.parquet o gs://)")
    ap.add_argument("--output_path", required=True, help="Ruta destino (.parquet o gs://)")
    ap.add_argument("--years", type=int, default=5, help="Ventana de años a conservar (default 5)")
    args = ap.parse_args()

    log.info(f"⚙️  Iniciando prepare_opt_data (window {args.years} años)")

    # 1) Carga
    df = _read_parquet(args.input_path)
    log.info(f"✔  Dataset cargado: {len(df):,} filas, {df.columns.tolist()} columnas")

    # 2) Asegurar nombres estándar
    rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "t": "timestamp"}
    missing = []
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    for col in ESSENTIAL:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Columnas esenciales ausentes después del rename: {missing}")

    # 3) Timestamp → datetime UTC
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        # Polygon entrega milisegundos → unit="ms"
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    # uniformizar a UTC
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # 4) Drop NaNs en OHLC
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        raise ValueError("Todos los registros contienen NaNs en OHLC; abortando.")

    # 5) Filtro temporal
    end_ts = df["timestamp"].max()
    start_ts = end_ts - pd.DateOffset(years=args.years)
    df_filtered = df[df["timestamp"] >= start_ts].copy()
    log.info(
        f"🗂  Filtrado desde {start_ts.date()} hasta {end_ts.date()} ⇒ "
        f"{len(df_filtered):,}/{len(df):,} filas"
    )

    if df_filtered.empty:
        raise ValueError("El filtro temporal devolvió 0 filas. Revisa las fechas / datos.")

    # 6) Guardar
    _write_parquet(df_filtered, args.output_path)
    log.info(f"✅  Parquet filtrado guardado en {args.output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"❌  prepare_opt_data terminó con error: {e}", exc_info=True)
        sys.exit(1)
