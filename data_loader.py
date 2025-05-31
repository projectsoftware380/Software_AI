#!/usr/bin/env python3
# ───────────────────────────── core/data_loader.py ───────────────────────────
"""
Carga un `.parquet`, aplica (opcionalmente) un scaler ``joblib`` y guarda la
versión escalada si se pasa `--upload`.

• Detecta modo GCP gracias a la variable de entorno `GOOGLE_CLOUD_PROJECT`.
• Rutas:
    - GCP  :  gs://<GCS_BUCKET>/data/{symbol}/{tf}/{file}.parquet
              gs://<GCS_BUCKET>/data_scaled/{symbol}/{tf}/{file}_scaled.parquet
    - Local:  ./data/{symbol}/{tf}/{file}.parquet
              ./data/{symbol}/{tf}/{file}_scaled.parquet
"""

# ────────────────────────── Imports & dependencias ──────────────────────────
import os, sys, tempfile, argparse, warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib

# instalamos google-cloud-storage si hace falta (para entornos locales)
try:
    from google.cloud import storage
except ImportError:                         # pragma: no cover
    import subprocess, importlib
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "google-cloud-storage>=2.16.0"],
        check=True,
    )
    from google.cloud import storage       # noqa: E402

# ─────────────────────────── Configuración General ──────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
LOCAL_DIR   = ROOT / "data"
GCS_BUCKET  = os.getenv("GCS_BUCKET", "trading-models-manuel")
GCP_MODE    = bool(os.getenv("GOOGLE_CLOUD_PROJECT"))

pd.options.mode.copy_on_write = True        # pequeña optimización
warnings.simplefilter("ignore", category=FutureWarning)

# ────────────────────────── Lectura / escritura segura ──────────────────────
def _read_parquet(path: str) -> pd.DataFrame:
    """
    Lee Parquet local o gs:// utilizando gcsfs (storage_options={'token':'cloud'})
    para evitar el requisito de GOOGLE_APPLICATION_CREDENTIALS dentro de Cloud Build.
    """
    if path.startswith("gs://"):
        return pd.read_parquet(path, engine="pyarrow",
                               storage_options={"token": "cloud"})
    return pd.read_parquet(path, engine="pyarrow")


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Escribe Parquet local o gs:// utilizando gcsfs con autenticación implícita.
    """
    if path.startswith("gs://"):
        df.to_parquet(path, engine="pyarrow",
                      storage_options={"token": "cloud"})
    else:
        df.to_parquet(path, engine="pyarrow")

# ──────────────────────────────── Helpers GCS ───────────────────────────────
def _gcs_client() -> storage.Client:
    """
    Crea un cliente de GCS, usando las credenciales definidas en
    GOOGLE_APPLICATION_CREDENTIALS si están presentes.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )
        return storage.Client(credentials=creds)
    return storage.Client()

def _upload_to_gcs(local_path: Path, gcs_uri: str) -> None:
    """
    Se mantiene por compatibilidad con llamadas externas.  Para el flujo
    principal ya no se utiliza porque `_write_parquet` envía directo a GCS.
    """
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = _gcs_client().bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"☁️  Subido a {gcs_uri}")

# ──────────────────────────────── Core Logic ───────────────────────────────
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

def load_and_scale(
    symbol: str,
    timeframe: str,
    filename: str,
    scaler_path: Optional[str] = None,
    upload: bool = False,
) -> pd.DataFrame:
    """
    Carga Parquet, aplica scaler y (opcionalmente) guarda la versión escalada.
    """
    in_path = _build_in_path(symbol, timeframe, filename)
    try:
        df = _read_parquet(in_path)
    except Exception as exc:
        raise RuntimeError(f"Error leyendo Parquet desde '{in_path}': {exc}") from exc

    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar scaler '{scaler_path}': {exc}") from exc

        try:
            scaled_arr = scaler.transform(df.values)
            df = pd.DataFrame(scaled_arr, columns=df.columns, index=df.index)
        except Exception as exc:
            raise RuntimeError(f"Error aplicando scaler: {exc}") from exc

    # Guardar si --upload
    if upload:
        out_path = _build_out_path(symbol, timeframe, filename)
        if GCP_MODE:
            # Escribimos directamente a GCS (usa gcsfs)
            _write_parquet(df, out_path)
        else:
            out_file = Path(out_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            _write_parquet(df, str(out_file))
            print(f"💾 Guardado local en {out_file}")
    return df

# ──────────────────────────── CLI de utilidad ──────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Carga y (opcional) escala datos Parquet.")
    p.add_argument("symbol",    help="Símbolo e.g. EURUSD")
    p.add_argument("timeframe", help="Timeframe e.g. 15minute")
    p.add_argument("filename",  help="Nombre base del archivo Parquet")
    p.add_argument("--scaler-path", type=str, default=None,
                   help="Ruta a scaler.joblib (opcional)")
    p.add_argument("--upload", action="store_true",
                   help="Guardar versión escalada (GCS/local)")
    args = p.parse_args()

    try:
        df = load_and_scale(args.symbol, args.timeframe, args.filename,
                            scaler_path=args.scaler_path, upload=args.upload)
        print(f"✅ Cargados {len(df):,} registros de '{args.filename}' "
              f"({ 'GCS' if GCP_MODE else 'LOCAL' })")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
