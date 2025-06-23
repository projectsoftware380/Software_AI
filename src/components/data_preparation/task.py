# src/components/data_preparation/task.py
"""
Tarea del componente de Preparación de Datos (con Logging Robusto).

Este script orquesta la carga de datos crudos, su limpieza, resampling,
la creación de un conjunto de hold-out para validación final, y la subida
de los artefactos resultantes a Google Cloud Storage.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import tempfile

import pandas as pd
from google.cloud import storage

# Módulos internos
from src.shared import constants, gcs_utils

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Funciones de Utilidad de Datos (Lógica Original Intacta)
# -----------------------------------------------------------------------------

def clean_and_resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = str(timeframe)
    if not isinstance(timeframe, str):
        raise TypeError(f"The 'timeframe' argument must be a string, but got {type(timeframe)} instead.")
    required_columns = {"open", "high", "low", "close", "volume"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {required_columns}")
    df.index.name = "timestamp"
    df = df.sort_index()
    df.dropna(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    return df


def create_holdout_set(df: pd.DataFrame, holdout_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El índice del DataFrame debe ser de tipo DatetimeIndex.")
    holdout_months = int(holdout_months)
    if holdout_months <= 0:
        return df, pd.DataFrame()
    holdout_start_date = df.index.max() - pd.DateOffset(months=holdout_months)
    train_df = df[df.index < holdout_start_date]
    holdout_df = df[df.index >= holdout_start_date]
    return train_df, holdout_df

def upload_df_to_gcs_and_verify(df: pd.DataFrame, gcs_uri: str) -> None:
    # Esta función ya tiene buen logging, se mantiene como está.
    logger.info(f"Intentando subir DataFrame a {gcs_uri}...")
    try:
        df.to_parquet(gcs_uri, engine="pyarrow", index=True)
        logger.info(f"La operación de escritura a {gcs_uri} se completó sin errores.")
        logger.info(f"Verificando la existencia del objeto en GCS: {gcs_uri}")
        client = storage.Client()
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"¡VERIFICACIÓN FALLIDA! El objeto no se encontró en GCS después de la subida: {gcs_uri}")
        logger.info(f"✅ VERIFICACIÓN EXITOSA: El objeto existe en {gcs_uri}.")
    except Exception as e:
        logger.error(f"Error fatal durante la subida o verificación a GCS: {e}", exc_info=True)
        raise

# -----------------------------------------------------------------------------
# Lógica Principal del Componente
# -----------------------------------------------------------------------------

def run_data_preparation(
    pair: str,
    timeframe: str,
    years_to_keep: int,
    holdout_months: int,
    cleanup: bool,
    prepared_data_path_output: str,
    holdout_data_path_output: str,
):
    """Orquesta todo el proceso de preparación de datos."""
    
    # [LOG] Punto de control inicial con todos los parámetros recibidos.
    logger.info(f"▶️ Iniciando data_preparation para el par '{pair}' con los siguientes parámetros:")
    logger.info(f"  - Timeframe: {timeframe}")
    logger.info(f"  - Años a mantener: {years_to_keep}")
    logger.info(f"  - Meses de Holdout: {holdout_months}")
    logger.info(f"  - Limpieza activada: {cleanup}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # [LOG] Registrar la ruta exacta que se va a buscar.
            gcs_input_uri = f"gs://{constants.BUCKET_NAME}/{constants.RAW_DATA_PATH}/{timeframe}/{pair}.parquet"
            logger.info(f"Intentando descargar datos crudos desde: {gcs_input_uri}")
            
            local_raw_path = gcs_utils.download_gcs_file(gcs_input_uri, tmp_path)
            
            if local_raw_path is None:
                raise FileNotFoundError(f"No se pudo descargar el archivo de datos crudos desde: {gcs_input_uri}")
            
            logger.info(f"✅ Datos crudos descargados exitosamente a: {local_raw_path}")
            
            df_raw = pd.read_parquet(local_raw_path)
            logger.info(f"DataFrame crudo cargado. Shape: {df_raw.shape}")
            
            df_processed = clean_and_resample(df_raw, timeframe)
            logger.info(f"DataFrame procesado y limpiado. Shape: {df_processed.shape}")
            
            df_train, df_holdout = create_holdout_set(df_processed, holdout_months)
            logger.info(f"División de datos completada. Shape entrenamiento: {df_train.shape}, Shape hold-out: {df_holdout.shape}")
            
            version_ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            output_base_path = f"gs://{constants.BUCKET_NAME}/{constants.FEATURES_PATH}/{pair}/{timeframe}"
            versioned_output_dir = f"{output_base_path}/{version_ts}"
            
            prepared_data_path = f"{versioned_output_dir}/{pair}_{timeframe}_train_opt.parquet"
            holdout_data_path = f"{versioned_output_dir}/{pair}_{timeframe}_holdout.parquet"
            
            logger.info(f"Ruta de salida para datos de entrenamiento: {prepared_data_path}")
            logger.info(f"Ruta de salida para datos de hold-out: {holdout_data_path}")
            
            upload_df_to_gcs_and_verify(df_train, prepared_data_path)
            upload_df_to_gcs_and_verify(df_holdout, holdout_data_path)
            
            Path(prepared_data_path_output).parent.mkdir(parents=True, exist_ok=True)
            Path(prepared_data_path_output).write_text(prepared_data_path)
            Path(holdout_data_path_output).parent.mkdir(parents=True, exist_ok=True)
            Path(holdout_data_path_output).write_text(holdout_data_path)
            logger.info("Rutas de salida escritas exitosamente para KFP.")
            
            if cleanup:
                logger.info(f"Iniciando limpieza de versiones antiguas en: {output_base_path}")
                gcs_utils.keep_only_latest_version(output_base_path)

    except Exception as e:
        # [LOG] Bloque de captura de errores con contexto completo.
        logger.critical(f"❌ Fallo fatal en el componente data_preparation para el par '{pair}'. Error: {e}", exc_info=True)
        raise

    logger.info(f"🏁 Componente data_preparation para '{pair}' completado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara datos para entrenamiento y backtesting.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--years-to-keep", type=int, default=5)
    parser.add_argument("--holdout-months", type=int, default=3)
    parser.add_argument("--cleanup", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--prepared-data-path-output", required=True)
    parser.add_argument("--holdout-data-path-output", required=True)
    
    args = parser.parse_args()
    
    # [LOG] Registro de los argumentos recibidos al iniciar el script.
    logger.info("Componente 'data_preparation' iniciado con los siguientes argumentos:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
    
    run_data_preparation(
        pair=args.pair,
        timeframe=args.timeframe,
        years_to_keep=args.years_to_keep,
        holdout_months=args.holdout_months,
        cleanup=args.cleanup,
        prepared_data_path_output=args.prepared_data_path_output,
        holdout_data_path_output=args.holdout_data_path_output,
    )