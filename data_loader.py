#!/usr/bin/env python3
# ─────────────────────────── core/data_loader.py ────────────────────────────
"""
Carga un Parquet (local o gs://), aplica opcionalmente un scaler ``joblib`` y
guarda la versión escalada si se indica ``--upload``.

Características
---------------
* Detecta modo GCP mediante ``GOOGLE_CLOUD_PROJECT``.
* Gestiona credenciales de GCS de forma implícita (``gcsfs token="cloud"``).
* Escala **solo columnas numéricas** (ignora timestamps u otras categóricas).
* Bucket por defecto: ``trading-ai-models-460823`` (sobrescribible con
  ``GCS_BUCKET``).

Rutas:
  • GCP  : gs://<bucket>/data/{symbol}/{tf}/{file}.parquet  
           gs://<bucket>/data_scaled/{symbol}/{tf}/{file}_scaled.parquet  
  • Local: ./data/{symbol}/{tf}/{file}.parquet  
           ./data/{symbol}/{tf}/{file}_scaled.parquet
"""

# ────────────────────────── Imports & dependencias ──────────────────────────
from __future__ import annotations
import os
import sys
import warnings
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib

# Instalar google-cloud-storage si el entorno local lo necesita
try:
    from google.cloud import storage
except ImportError:                           # pragma: no cover
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "google-cloud-storage>=2.16.0"],
        check=True,
    )
    from google.cloud import storage         # noqa: E402

# ─────────────────────────── Configuración global ───────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
LOCAL_DIR   = ROOT / "data"
GCS_BUCKET  = os.getenv("GCS_BUCKET", "trading-ai-models-460823")
GCP_MODE    = bool(os.getenv("GOOGLE_CLOUD_PROJECT"))

pd.options.mode.copy_on_write = True
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ────────────────────────────── I/O helpers ─────────────────────────────────
def _read_parquet(path: str) -> pd.DataFrame:
    """Lee Parquet tanto local como gs:// usando gcsfs en modo ‘token=cloud’."""
    opts = {"engine": "pyarrow"}
    if path.startswith("gs://"):
        opts["storage_options"] = {"token": "cloud"}
    return pd.read_parquet(path, **opts)


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    """Escribe Parquet local o gs:// con gcsfs y autenticación implícita."""
    opts = {"engine": "pyarrow"}
    if path.startswith("gs://"):
        opts["storage_options"] = {"token": "cloud"}
    df.to_parquet(path, index=False, **opts)

# ──────────────────────────── Path builders ────────────────────────────────
def _build_in_path(symbol: str, tf: str, fname: str) -> str:
    fn = f"{fname}.parquet"
    if GCP_MODE:
        return f"gs://{GCS_BUCKET}/data/{symbol}/{tf}/{fn}"
    return str(LOCAL_DIR / symbol / tf / fn)


def _build_out_path(symbol: str, tf: str, fname: str) -> str:
    fn = f"{fname}_scaled.parquet"
    if GCP_MODE:
        return f"gs://{GCS_BUCKET}/data_scaled/{symbol}/{tf}/{fn}"
    return str(LOCAL_DIR / symbol / tf / fn)

# ──────────────────────────── Carga y escalado ─────────────────────────────
def load_and_scale(
    symbol: str,
    timeframe: str,
    filename: str,
    scaler_path: Optional[str] = None,
    upload: bool = False,
) -> pd.DataFrame:
    """Carga Parquet, aplica scaler a columnas numéricas y guarda si se solicita."""
    in_path = _build_in_path(symbol, timeframe, filename)

    try:
        df = _read_parquet(in_path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Error leyendo Parquet desde '{in_path}': {exc}") from exc

    # Normalizar timestamp si existe
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ---------------- Escalado opcional ------------------------------------
    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"No se pudo cargar scaler '{scaler_path}': {exc}") from exc

        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            raise ValueError("No se encontraron columnas numéricas para escalar.")

        try:
            scaled_arr = scaler.transform(df[num_cols].values)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Error aplicando scaler: {exc}") from exc

        df[num_cols] = scaled_arr  # Solo sustituimos las numéricas; el resto queda igual

    # ---------------- Guardado opcional ------------------------------------
    if upload:
        out_path = _build_out_path(symbol, timeframe, filename)
        # Crear carpetas locales si corresponde
        if not out_path.startswith("gs://"):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, out_path)
        location = "GCS" if out_path.startswith("gs://") else "LOCAL"
        print(f"💾 Datos escalados guardados en {out_path} ({location})")

    return df

# ───────────────────────────── CLI utilitario ──────────────────────────────
if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Carga y (opcionalmente) escala un Parquet.")
    cli.add_argument("symbol",    help="Símbolo (p. ej. EURUSD)")
    cli.add_argument("timeframe", help="Timeframe (p. ej. 15minute)")
    cli.add_argument("filename",  help="Nombre base del archivo Parquet (sin extensión)")
    cli.add_argument("--scaler-path", type=str, default=None,
                     help="Ruta a scaler.joblib (opcional)")
    cli.add_argument("--upload", action="store_true",
                     help="Guardar la versión escalada")
    args = cli.parse_args()

    try:
        df_loaded = load_and_scale(
            args.symbol,
            args.timeframe,
            args.filename,
            scaler_path=args.scaler_path,
            upload=args.upload,
        )
        src = "GCS" if GCP_MODE else "LOCAL"
        print(f"✅ {len(df_loaded):,} registros cargados desde {src}.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
