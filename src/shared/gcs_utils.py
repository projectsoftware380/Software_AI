# src/shared/gcs_utils.py
"""
Módulo de utilidades para interactuar con Google Cloud Storage (GCS).

Centraliza toda la lógica para leer, escribir y manipular objetos en GCS,
evitando la duplicación de código en los diferentes componentes de la pipeline.

Funciones:
- `get_gcs_client()`: Retorna un cliente de GCS autenticado.
- `upload_gcs_file()`: Sube un archivo local a una URI de GCS.
- `download_gcs_file()`: Descarga un objeto de GCS a un directorio local.
- `copy_gcs_object()`: Copia un objeto entre dos ubicaciones dentro de GCS.
- `gcs_path_exists()`: Verifica si una ruta (blob) de GCS existe.
- `delete_gcs_blob()`: Elimina un objeto específico de GCS.
- `ensure_gcs_path_and_get_local()`: Descarga un archivo si es de GCS;
  de lo contrario, devuelve la ruta local.
- `list_gcs_files()`: Lista archivos en un prefijo GCS.
- `find_latest_gcs_file_in_timestamped_dirs()`: Encuentra la última versión de un archivo en directorios con timestamp.
"""

from __future__ import annotations

import logging
import os
import re # Necesario para expresiones regulares
import tempfile
from pathlib import Path
from typing import List

from google.api_core import exceptions
from google.cloud import storage
from google.oauth2 import service_account
import gcsfs # Importar gcsfs para listar directorios

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)


def get_gcs_client() -> storage.Client:
    """
    Retorna un cliente de GCS autenticado de forma robusta.

    Intenta usar las credenciales de la variable de entorno
    `GOOGLE_APPLICATION_CREDENTIALS`. Si no está definida o el archivo no
    existe, recurre a las Credenciales por Defecto de la Aplicación (ADC),
    común en entornos de GCP como Vertex AI.
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        try:
            creds = service_account.Credentials.from_service_account_file(creds_path)
            return storage.Client(credentials=creds)
        except Exception as e:
            logger.warning(
                f"No se pudieron usar las credenciales de '{creds_path}': {e}. "
                "Intentando con credenciales por defecto."
            )
    
    # Si lo anterior falla o no se especifica, usa ADC
    return storage.Client()


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Valida y descompone una URI de GCS en (bucket, blob_name)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"La URI proporcionada no es una ruta GCS válida: {uri}")
    parts = uri[5:].split("/", 1)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"La URI de GCS está mal formada: {uri}")
    return parts[0], parts[1]


def upload_gcs_file(local_path: Path, gcs_uri: str) -> None:
    """
    Sube un archivo local a una ubicación en GCS.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"El archivo local a subir no existe: {local_path}")

    try:
        client = get_gcs_client()
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        logger.info(f"☁️  Subiendo {local_path.name} a {gcs_uri}...")
        blob.upload_from_filename(str(local_path))
        logger.info(f"✔  Subida completada: {gcs_uri}")
    except Exception as e:
        logger.error(f"❌ Falló la subida de {local_path} a {gcs_uri}: {e}")
        raise


def download_gcs_file(gcs_uri: str, destination_dir: Path | None = None) -> Path:
    """
    Descarga un archivo de GCS a un directorio local.

    Si `destination_dir` es None, se crea un directorio temporal.
    Retorna la ruta local del archivo descargado.
    """
    dest_dir = destination_dir or Path(tempfile.mkdtemp())
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        client = get_gcs_client()
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        
        local_path = dest_dir / Path(blob_name).name
        
        blob = client.bucket(bucket_name).blob(blob_name)
        logger.info(f"📥 Descargando {gcs_uri} a {local_path}...")
        blob.download_to_filename(local_path)
        logger.info(f"✔  Descarga completada: {local_path}")
        return local_path
    except exceptions.NotFound:
        logger.error(f"❌ El objeto no existe en GCS: {gcs_uri}")
        raise
    except Exception as e:
        logger.error(f"❌ Falló la descarga de {gcs_uri}: {e}")
        raise


def copy_gcs_object(source_uri: str, destination_uri: str) -> None:
    """Copia un objeto de una ubicación GCS a otra."""
    try:
        client = get_gcs_client()
        src_bucket_name, src_blob_name = _parse_gcs_uri(source_uri)
        dst_bucket_name, dst_blob_name = _parse_gcs_uri(destination_uri)

        source_bucket = client.bucket(src_bucket_name)
        source_blob = source_bucket.blob(src_blob_name)
        destination_bucket = client.bucket(dst_bucket_name)

        logger.info(f"🔄 Copiando en GCS de {source_uri} a {destination_uri}...")
        source_bucket.copy_blob(source_blob, destination_bucket, dst_blob_name)
        logger.info("✔  Copia completada.")
    except exceptions.NotFound:
        logger.error(f"❌ El objeto fuente no existe para copiar: {source_uri}")
        raise
    except Exception as e:
        logger.error(f"❌ Falló la copia en GCS: {e}")
        raise


def gcs_path_exists(gcs_uri: str) -> bool:
    """Verifica si un blob existe en GCS."""
    try:
        client = get_gcs_client()
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        return client.bucket(bucket_name).blob(blob_name).exists()
    except Exception as e:
        logger.warning(f"No se pudo verificar la existencia de {gcs_uri}: {e}")
        return False


def delete_gcs_blob(gcs_uri: str) -> None:
    """Elimina un blob específico de GCS si existe."""
    try:
        client = get_gcs_client()
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        blob = client.bucket(bucket_name).blob(blob_name)
        if blob.exists():
            logger.info(f"🗑️  Eliminando blob: {gcs_uri}...")
            blob.delete()
            logger.info("✔  Blob eliminado.")
        else:
            logger.info(f"ℹ️  El blob a eliminar no existía: {gcs_uri}")
    except Exception as e:
        logger.error(f"❌ Falló la eliminación del blob {gcs_uri}: {e}")
        raise


def ensure_gcs_path_and_get_local(path_or_uri: str) -> Path:
    """
    Función de conveniencia. Si la ruta es de GCS, la descarga a una
    carpeta temporal y retorna la ruta local. Si ya es local, solo
    verifica que exista y la retorna como un objeto Path.
    """
    if path_or_uri.startswith("gs://"):
        return download_gcs_file(path_or_uri)
    
    local_path = Path(path_or_uri)
    if not local_path.exists():
        raise FileNotFoundError(f"La ruta local especificada no existe: {local_path}")
    return local_path

def list_gcs_files(gcs_prefix: str, suffix: str = "") -> List[str]:
    """
    Lista todos los archivos dentro de un prefijo de GCS, opcionalmente filtrando por sufijo.

    Args:
        gcs_prefix (str): La ruta GCS (ej. 'gs://your-bucket/your-folder').
        suffix (str): Sufijo opcional para filtrar archivos (ej. '.parquet').

    Returns:
        List[str]: Una lista de URIs GCS de los archivos encontrados.
    """
    if not gcs_prefix.startswith("gs://"):
        raise ValueError(f"El prefijo GCS debe comenzar con 'gs://': {gcs_prefix}")

    # Es importante usar el project_id correcto si no está en la variable de entorno
    # Usaremos os.getenv("GOOGLE_CLOUD_PROJECT") o una constante como constants.PROJECT_ID
    # Para este ejemplo, asumiremos que GOOGLE_CLOUD_PROJECT está configurado o lo manejamos.
    fs = gcsfs.GCSFileSystem(project=os.getenv("GOOGLE_CLOUD_PROJECT")) 
    
    # Asegurarse de que el prefijo termina en '/' para listar contenido de carpeta
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"

    files = []
    try:
        # fs.ls devuelve rutas relativas al bucket, ej. 'bucket-name/folder/file.txt'
        # Necesitamos reconstruir la URI completa 'gs://bucket-name/folder/file.txt'
        bucket_name = gcs_prefix.split("gs://")[1].split("/")[0]
        for path in fs.ls(gcs_prefix, detail=False):
            if fs.isfile(path) and path.endswith(suffix):
                files.append(f"gs://{path}")
    except Exception as e:
        logger.error(f"❌ Error al listar archivos en GCS en {gcs_prefix}: {e}")
        raise
        
    return files

def find_latest_gcs_file_in_timestamped_dirs(base_gcs_path: str, filename: str) -> str | None:
    """
    Encuentra la ruta completa de la última versión de un archivo
    dentro de un conjunto de subdirectorios con marca de tiempo.

    Ejemplo: busca 'best_params.json' en 'gs://bucket/params/pair/YYYYMMDDHHMMSS/'

    Args:
        base_gcs_path (str): La ruta base que contiene los directorios con timestamp (ej. 'gs://bucket/params/LSTM_v3/EURUSD/').
        filename (str): El nombre del archivo a buscar (ej. 'best_params.json').

    Returns:
        str | None: La URI completa del archivo más reciente, o None si no se encuentra.
    """
    if not base_gcs_path.startswith("gs://"):
        raise ValueError(f"La ruta base GCS debe comenzar con 'gs://': {base_gcs_path}")

    fs = gcsfs.GCSFileSystem(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    
    # Asegurarse de que el prefijo termina en '/'
    if not base_gcs_path.endswith("/"):
        base_gcs_path += "/"

    timestamp_dirs = []
    try:
        # Listar subdirectorios que coincidan con el patrón de marca de tiempo (YYYYMMDDHHMMSS)
        # fs.ls devuelve rutas relativas al bucket, ej. 'bucket/path/timestamp_dir'
        for path in fs.ls(base_gcs_path, detail=False):
            dir_name = Path(path).name # Obtener solo el nombre del directorio (la marca de tiempo)
            if re.fullmatch(r'\d{14}', dir_name) and fs.isdir(path):
                timestamp_dirs.append(path)
    except Exception as e:
        logger.warning(f"No se pudieron listar directorios en {base_gcs_path}: {e}")
        return None

    if not timestamp_dirs:
        logger.warning(f"No se encontraron subdirectorios con marca de tiempo en {base_gcs_path}")
        return None

    # Ordenar los directorios por timestamp (el más reciente al final)
    timestamp_dirs.sort()

    # Construir la ruta al archivo en el directorio más reciente
    latest_dir_path = timestamp_dirs[-1]
    full_file_path_relative = f"{latest_dir_path}/{filename}"
    
    # Reconstruir la URI GCS completa
    bucket_name = base_gcs_path.split("gs://")[1].split("/")[0]
    final_gcs_uri = f"gs://{full_file_path_relative}"

    if gcs_path_exists(final_gcs_uri):
        logger.info(f"✔ Encontrado el archivo más reciente: {final_gcs_uri}")
        return final_gcs_uri
    else:
        logger.warning(f"El archivo {filename} no se encontró en la última carpeta con timestamp: {final_gcs_uri}")
        return None