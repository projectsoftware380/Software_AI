# -----------------------------------------------------------------------------
# task.py: Componente de Preparación de Datos
# -----------------------------------------------------------------------------
# Este script orquesta la carga de datos crudos, su limpieza, resampling,
# la creación de un conjunto de hold-out para validación final, y la subida
# de los artefactos resultantes (datasets de entrenamiento y hold-out) a
# Google Cloud Storage.
# -----------------------------------------------------------------------------

import argparse
import logging
from datetime import datetime
from pathlib import Path
import tempfile

import pandas as pd
from google.cloud import storage

# Módulos internos
from src.shared import constants, gcs_utils

# Configuración del logging para una salida clara y estandarizada
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------------------------------------------------------
# Funciones de Utilidad de Datos (Mantenidas dentro de este script)
# -----------------------------------------------------------------------------

def clean_and_resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Limpia y remuestrea un DataFrame de precios OHLCV.
    """
    # --- AJUSTE CLAVE AÑADIDO ---
    # Se convierte el parámetro 'timeframe' a string para asegurar compatibilidad
    # con el orquestador de pipelines (KFP), que puede pasar objetos placeholder.
    timeframe = str(timeframe)

    if not isinstance(timeframe, str):
        raise TypeError(
            f"The 'timeframe' argument must be a string, but got {type(timeframe)} instead."
        )

    required_columns = {"open", "high", "low", "close", "volume"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"El DataFrame debe contener las columnas: {required_columns}"
        )

    df.index.name = "timestamp"
    df = df.sort_index()
    df.dropna(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    logging.info(f"DataFrame limpiado. {df.shape[0]} filas restantes.")
    return df


def create_holdout_set(df: pd.DataFrame, holdout_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el DataFrame en un conjunto de entrenamiento y uno de hold-out.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El índice del DataFrame debe ser de tipo DatetimeIndex.")

    # Convertir holdout_months a entero por si KFP lo pasa como otro tipo
    holdout_months = int(holdout_months)
    
    if holdout_months <= 0:
        logging.warning("holdout_months es <= 0. No se creará un set de hold-out.")
        return df, pd.DataFrame()

    holdout_start_date = df.index.max() - pd.DateOffset(months=holdout_months)
    train_df = df[df.index < holdout_start_date]
    holdout_df = df[df.index >= holdout_start_date]

    logging.info(f"Holdout creado a partir de {holdout_start_date.date()}")
    return train_df, holdout_df

# -----------------------------------------------------------------------------
# Lógica Principal del Componente
# -----------------------------------------------------------------------------

def upload_df_to_gcs_and_verify(df: pd.DataFrame, gcs_uri: str) -> None:
    """Sube un DataFrame a una URI de GCS y verifica la subida."""
    logging.info(f"Intentando subir DataFrame a {gcs_uri}...")
    try:
        df.to_parquet(gcs_uri, engine="pyarrow", index=True)
        logging.info(
            f"La operación de escritura a {gcs_uri} se completó sin errores."
        )
        logging.info(f"Verificando la existencia del objeto en GCS: {gcs_uri}")
        client = storage.Client()
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(
                f"¡VERIFICACIÓN FALLIDA! El objeto no se encontró en GCS "
                f"después de la subida: {gcs_uri}"
            )
        logging.info(f"✅ VERIFICACIÓN EXITOSA: El objeto existe en {gcs_uri}.")
    except Exception as e:
        logging.error(f"Error fatal durante la subida o verificación a GCS: {e}")
        raise


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
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # La ruta al archivo de entrada es construida dinámicamente.
        gcs_input_uri = f"gs://{constants.BUCKET_NAME}/{constants.RAW_DATA_PATH}/{timeframe}/{pair}.parquet"
        
        local_raw_path = gcs_utils.download_gcs_file(gcs_input_uri, tmp_path)
        
        if local_raw_path is None:
            raise FileNotFoundError(f"No se pudo descargar el archivo de datos crudos desde: {gcs_input_uri}")
            
        df_raw = pd.read_parquet(local_raw_path)
        df_processed = clean_and_resample(df_raw, timeframe)
        df_train, df_holdout = create_holdout_set(df_processed, holdout_months)
        
        logging.info(
            f"Set de entrenamiento: {df_train.shape[0]} filas | "
            f"Set de Hold-out: {df_holdout.shape[0]} filas"
        )
        
        version_ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        output_base_path = f"gs://{constants.BUCKET_NAME}/{constants.FEATURES_PATH}/{pair}/{timeframe}/{version_ts}"
        
        prepared_data_path = f"{output_base_path}/{pair}_{timeframe}_train_opt.parquet"
        holdout_data_path = f"{output_base_path}/{pair}_{timeframe}_holdout.parquet"
        
        upload_df_to_gcs_and_verify(df_train, prepared_data_path)
        upload_df_to_gcs_and_verify(df_holdout, holdout_data_path)
        
        Path(prepared_data_path_output).parent.mkdir(parents=True, exist_ok=True)
        Path(prepared_data_path_output).write_text(prepared_data_path)
        
        Path(holdout_data_path_output).parent.mkdir(parents=True, exist_ok=True)
        Path(holdout_data_path_output).write_text(holdout_data_path)
        
        logging.info(
            "Rutas de salida escritas para KFP: "
            f"Train='{prepared_data_path}', Holdout='{holdout_data_path}'"
        )
        
        if cleanup:
            logging.info("La lógica de limpieza de versiones antiguas no está implementada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara datos para entrenamiento y backtesting."
    )
    parser.add_argument("--pair", required=True, help="Par de divisas a procesar.")
    parser.add_argument("--timeframe", required=True, help="Timeframe de los datos.")
    parser.add_argument(
        "--years-to-keep",
        type=int,
        default=5,
        help="Años de datos a mantener antes del holdout.",
    )
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=3,
        help="Meses para reservar para el set de hold-out.",
    )
    # El tipo `bool` para argparse es complicado; se recomienda manejarlo así
    parser.add_argument(
        "--cleanup",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Activar limpieza de versiones antiguas."
    )
    parser.add_argument(
        "--prepared-data-path-output",
        required=True,
        help="Ruta donde guardar la URI del dataset de entrenamiento.",
    )
    parser.add_argument(
        "--holdout-data-path-output",
        required=True,
        help="Ruta donde guardar la URI del dataset de hold-out.",
    )
    args = parser.parse_args()
    
    run_data_preparation(
        pair=args.pair,
        timeframe=args.timeframe,
        years_to_keep=args.years_to_keep,
        holdout_months=args.holdout_months,
        cleanup=args.cleanup,
        prepared_data_path_output=args.prepared_data_path_output,
        holdout_data_path_output=args.holdout_data_path_output,
    )