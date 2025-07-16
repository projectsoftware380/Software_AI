# src/components/data_preparation/task.py
"""
Tarea del componente de preparación de datos.
Puede funcionar en modo KFP (leyendo localmente) o en modo local (leyendo/escribiendo en GCS).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from google.cloud import storage
import gcsfs # Importar gcsfs

from src.shared.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"open", "high", "low", "close", "volume"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {required_columns}")
    df.index.name = "timestamp"
    df = df.sort_index().dropna()
    df = df[~df.index.duplicated(keep="first")]
    return df

def create_holdout_set(df: pd.DataFrame, holdout_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El índice debe ser de tipo DatetimeIndex.")
    if holdout_months <= 0:
        return df, pd.DataFrame()
    holdout_start_date = df.index.max() - pd.DateOffset(months=holdout_months)
    train_df = df[df.index < holdout_start_date]
    holdout_df = df[df.index >= holdout_start_date]
    return train_df, holdout_df

def save_df_to_gcs_and_verify(df: pd.DataFrame, gcs_path: str):
    logger.info(f"Guardando DataFrame en GCS: {gcs_path}")
    try:
        fs = gcsfs.GCSFileSystem() # Inicializar el sistema de archivos GCS
        df.to_parquet(gcs_path, index=True, filesystem=fs) # Usar el objeto fs
        storage_client = storage.Client()
        bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"¡VERIFICACIÓN FALLIDA! El archivo no se encontró en GCS: {gcs_path}")
        logger.info(f"✅ VERIFICACIÓN EXITOSA: El archivo existe en {gcs_path}.")
    except Exception as e:
        logger.error(f"Error al guardar o verificar el archivo en GCS: {e}", exc_info=True)
        raise

def run_data_preparation(
    years_to_keep: int,
    holdout_months: int,
    # Argumentos para diferentes modos
    input_data_path: str | None = None, # Modo local: Ruta GCS al archivo de entrada
    output_gcs_path: str | None = None, # Modo local: Ruta GCS base para salidas
    input_data_dir: str | None = None,  # Modo KFP: Directorio local de entrada
    prepared_data_path_output: str | None = None, # Modo KFP
    holdout_data_path_output: str | None = None,  # Modo KFP
):
    is_local_run = input_data_path is not None
    logger.info(f"▶️ Iniciando data_preparation. Modo de ejecución: {'Local' if is_local_run else 'KFP'}")

    try:
        fs = gcsfs.GCSFileSystem() # Inicializar el sistema de archivos GCS
        if is_local_run:
            logger.info(f"Leyendo datos directamente desde GCS: {input_data_path}")
            df_full = pd.read_parquet(input_data_path, filesystem=fs) # Usar el objeto fs
            # Extraer par y timeframe de la ruta GCS
            parts = input_data_path.split('/')
            timeframe = parts[-2]
            pair = parts[-3]
        else: # Modo KFP
            input_path = Path(input_data_dir)
            all_files = list(input_path.glob("*.parquet"))
            if not all_files:
                raise FileNotFoundError(f"No se encontraron archivos .parquet en: {input_data_dir}")
            logger.info(f"Cargando {all_files[0]} desde el directorio local.")
            df_full = pd.read_parquet(all_files[0])
            # Extraer par y timeframe del nombre del archivo
            file_name_parts = all_files[0].stem.split('_')
            pair = file_name_parts[1]
            timeframe = file_name_parts[2]

        if 'timestamp' in df_full.columns:
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
            df_full = df_full.set_index('timestamp')
        else:
            raise ValueError("La columna 'timestamp' no se encontró.")

        df_processed = clean_and_resample(df_full)
        df_train, df_holdout = create_holdout_set(df_processed, holdout_months)
        logger.info(f"División de datos completada. Entrenamiento: {df_train.shape}, Holdout: {df_holdout.shape}")

        # --- Guardado de resultados ---
        if is_local_run:
            gcs_base_path = f"{output_gcs_path}/{pair}/{timeframe}"
            prepared_gcs_path = f"{gcs_base_path}/train_opt/prepared_data.parquet"
            holdout_gcs_path = f"{gcs_base_path}/holdout/holdout_data.parquet"
        else: # Modo KFP
            bucket_name = prepared_data_path_output.split('/')[2]
            gcs_base_path = f"gs://{bucket_name}/prepared_data/{pair}/{timeframe}"
            prepared_gcs_path = f"{gcs_base_path}/train_opt.parquet"
            holdout_gcs_path = f"{gcs_base_path}/holdout.parquet"

        save_df_to_gcs_and_verify(df_train, prepared_gcs_path)
        save_df_to_gcs_and_verify(df_holdout, holdout_gcs_path)

        if not is_local_run:
            Path(prepared_data_path_output).write_text(prepared_gcs_path)
            Path(holdout_data_path_output).write_text(holdout_gcs_path)
            logger.info("Rutas de GCS de salida escritas para KFP.")

    except Exception as e:
        logger.exception(f"❌ Fallo fatal en data_preparation: {e}")
        sys.exit(1)

    logger.info("🏁 Componente data_preparation completado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara datos para el entrenamiento.")
    # Argumentos comunes
    parser.add_argument("--years-to-keep", type=int, default=10)
    parser.add_argument("--holdout-months", type=int, default=6)

    # Argumentos para modo local/híbrido
    parser.add_argument("--input-data-path", required=False, help="Ruta GCS al archivo parquet de entrada.")
    parser.add_argument("--output-gcs-path", required=False, help="Ruta GCS base para guardar los datos preparados.")

    # Argumentos para modo KFP
    parser.add_argument("--input-data-dir", required=False, help="Directorio local de entrada (provisto por KFP).")
    parser.add_argument("--prepared-data-path-output", required=False, help="Archivo de salida para la ruta GCS de datos de entrenamiento (KFP).")
    parser.add_argument("--holdout-data-path-output", required=False, help="Archivo de salida para la ruta GCS de datos de holdout (KFP).")

    args = parser.parse_args()

    if args.input_data_path: # Modo local
        if not args.output_gcs_path:
            raise ValueError("--output-gcs-path es requerido en modo local.")
    elif not all([args.input_data_dir, args.prepared_data_path_output, args.holdout_data_path_output]):
        raise ValueError("Se requieren los argumentos de KFP para la ejecución en modo pipeline.")

    logger.info("Componente 'data_preparation' iniciado con argumentos: %s", vars(args))
    run_data_preparation(**vars(args))