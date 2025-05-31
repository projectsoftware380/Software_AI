#!/usr/bin/env python3
"""
prepare_opt_data.py
────────────────────
Prepara el subset de datos históricos para la optimización de hiperparámetros.
Carga el parquet completo desde GCS, lo filtra para los últimos 5 años
y guarda el resultado en una nueva ubicación en GCS.

Uso:
    python prepare_opt_data.py --input_path gs://bucket/data/SYMBOL_TIMEFRAME.parquet \
                               --output_path gs://bucket/data_filtered_for_opt/SYMBOL_TIMEFRAME_recent.parquet
"""

import os
import sys # Asegúrate de que sys esté importado si lo usas en el try-except de abajo
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pandas as pd
import tempfile # <--- AÑADIDO IMPORT tempfile

# Importaciones para GCS
try:
    import gcsfs
    from google.cloud import storage
    # from google.oauth2 import service_account # No se usa directamente si usas credenciales por defecto o gcsfs con token cloud
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    # Esto es una medida de seguridad, en el Dockerfile ya deberían estar instalados
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gcsfs", "google-cloud-storage"])
    import gcsfs
    from google.cloud import storage
    # from google.oauth2 import service_account
    from google.auth.exceptions import DefaultCredentialsError


# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Constantes de Configuración ---
OPTIMIZATION_WINDOW_YEARS = 5

# --- Helpers GCS ---
def get_gcs_client():
    # Para entornos de GCP como Vertex AI, Cloud Run, etc., el cliente
    # de storage.Client() tomará las credenciales del entorno automáticamente.
    # No es necesario especificar credenciales si la cuenta de servicio del entorno
    # tiene los permisos adecuados.
    try:
        return storage.Client()
    except DefaultCredentialsError as e:
        logger.error(f"No se pudieron obtener credenciales predeterminadas de GCP: {e}.")
        logger.error("Asegúrate de que el entorno de ejecución (VM, pod, etc.) tenga una cuenta de servicio con permisos para GCS, o que GOOGLE_APPLICATION_CREDENTIALS esté configurado si se ejecuta localmente con una SA específica.")
        raise

def download_gs(uri: str) -> Path:
    client = get_gcs_client()
    if not uri.startswith("gs://"):
        raise ValueError(f"La URI proporcionada no es una ruta GCS válida: {uri}")
    
    try:
        bucket_name, blob_name = uri[5:].split("/", 1)
    except ValueError:
        raise ValueError(f"Formato de URI GCS inválido: {uri}. Debe ser gs://bucket-name/path/to/blob")

    # Crear un nombre de archivo local único en el directorio temporal
    local_file_path = Path(tempfile.mkdtemp()) / Path(blob_name).name
    
    try:
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.download_to_filename(local_file_path)
        logger.info(f"Archivo descargado de {uri} a {local_file_path}")
    except Exception as e:
        logger.error(f"Fallo al descargar {uri} a {local_file_path}: {e}")
        raise
    return local_file_path

def upload_gs(local_path: Path, gcs_uri: str):
    client = get_gcs_client()
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"La URI GCS proporcionada no es una ruta válida: {gcs_uri}")

    try:
        bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    except ValueError:
        raise ValueError(f"Formato de URI GCS inválido: {gcs_uri}. Debe ser gs://bucket-name/path/to/blob")
    
    try:
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Archivo subido de {local_path} a {gcs_uri}")
    except Exception as e:
        logger.error(f"Fallo al subir {local_path} a {gcs_uri}: {e}")
        raise

def save_dataframe_to_gcs(df: pd.DataFrame, gcs_path: str):
    if not gcs_path.startswith("gs://"):
        raise ValueError("La ruta de salida debe ser una ruta gs://")
    logger.info(f"Intentando guardar DataFrame en GCS: {gcs_path} usando gcsfs y token cloud.")
    try:
        # gcsfs usa las credenciales del entorno ("cloud")
        df.to_parquet(gcs_path, index=False, engine="pyarrow", storage_options={"token": "cloud"})
        logger.info(f"DataFrame guardado en GCS: {gcs_path} usando gcsfs.")
    except Exception as e:
        logger.error(f"Error guardando DataFrame a GCS con gcsfs: {e}", exc_info=True)
        raise


# --- Lógica Principal ---
def main():
    parser = argparse.ArgumentParser(description="Prepara datos para la optimización de hiperparámetros.")
    parser.add_argument("--input_path", required=True,
                        help="Ruta gs:// al archivo Parquet completo de datos históricos.")
    parser.add_argument("--output_path", required=True,
                        help="Ruta gs:// donde se guardará el archivo Parquet filtrado para la optimización.")
    args = parser.parse_args()

    logger.info(f"[{datetime.utcnow().isoformat()} UTC] Iniciando preparación de datos para optimización.")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Ventana de optimización: últimos {OPTIMIZATION_WINDOW_YEARS} años.")

    try:
        logger.info(f"Intentando leer Parquet directamente desde GCS: {args.input_path}")
        df_full = pd.read_parquet(args.input_path, engine="pyarrow", storage_options={"token": "cloud"})
        logger.info(f"Datos completos cargados: {len(df_full):,} filas.")

        if 'timestamp' not in df_full.columns:
            raise ValueError("La columna 'timestamp' no se encontró en el DataFrame de entrada.")
        
        # Asegurarse de que el timestamp sea datetime y esté localizado en UTC o sea naive
        # Si ya es datetime[ns, UTC], no se hace nada. Si es solo datetime[ns], se asume UTC.
        if not pd.api.types.is_datetime64_any_dtype(df_full['timestamp']):
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], unit='ms', errors='coerce')
        
        if df_full['timestamp'].dt.tz is None:
            df_full['timestamp'] = df_full['timestamp'].dt.tz_localize('UTC')
        else:
            df_full['timestamp'] = df_full['timestamp'].dt.tz_convert('UTC')


        end_date_for_filter = df_full['timestamp'].max()
        # Para restar años, es más seguro usar relativedelta si los datos son muy extensos,
        # pero para 5 años, timedelta es generalmente aceptable.
        start_date_for_filter = end_date_for_filter - pd.DateOffset(years=OPTIMIZATION_WINDOW_YEARS)
        
        logger.info(f"Fecha máxima en datos: {end_date_for_filter}")
        logger.info(f"Fecha de inicio calculada para el filtro: {start_date_for_filter}")

        df_filtered = df_full[df_full['timestamp'] >= start_date_for_filter].copy()
        
        if df_filtered.empty:
            logger.warning(f"DataFrame filtrado vacío para el período de {OPTIMIZATION_WINDOW_YEARS} años. Esto puede indicar falta de datos recientes o un rango incorrecto.")
            # Decide si quieres que esto sea un error fatal o no
            # raise ValueError("El DataFrame filtrado resultó vacío.")

        logger.info(f"Datos filtrados para optimización: {len(df_filtered):,} filas (desde {df_filtered['timestamp'].min()} hasta {df_filtered['timestamp'].max()}).")

        save_dataframe_to_gcs(df_filtered, args.output_path)
        logger.info(f"✅ Preparación de datos para optimización completada y guardada en {args.output_path}")

    except Exception as e:
        logger.critical(f"❌ Error durante la preparación de datos para optimización: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()