# src/shared/gcs_utils.py
"""
Módulo de utilidades para interactuar con Google Cloud Storage (GCS).

Este módulo centraliza todas las operaciones comunes de GCS para evitar la
duplicación de código y asegurar un manejo de errores consistente a través
de todos los componentes del pipeline.
"""
from __future__ import annotations

import logging
import re # <-- AJUSTE: Importación añadida para la nueva función
from pathlib import Path
import tempfile

from google.api_core import exceptions
from google.cloud import storage

# --- Configuración (Sin Cambios) ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

_GCS_CLIENT = None


def _get_gcs_client() -> storage.Client:
    """
    Inicializa y devuelve un cliente de GCS, reutilizando la instancia si ya existe.
    """
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        logger.info("Inicializando cliente de Google Cloud Storage.")
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def gcs_path_exists(gcs_uri: str) -> bool:
    """
    Verifica si un objeto (archivo) existe en una ruta de GCS.
    """
    # ... (Lógica original de esta función intacta)
    client = _get_gcs_client()
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error al verificar la existencia de '{gcs_uri}': {e}")
        return False


def upload_gcs_file(local_path: Path, gcs_uri: str) -> None:
    """
    Sube un archivo local a una ruta de GCS.
    """
    # ... (Lógica original de esta función intacta)
    if not gcs_path_exists(gcs_uri):
        logger.info(f"Subiendo archivo a GCS: {local_path} -> {gcs_uri}")
        client = _get_gcs_client()
        try:
            bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
        except Exception as e:
            logger.error(f"Fallo al subir archivo a '{gcs_uri}': {e}", exc_info=True)
            raise
    else:
        logger.info(f"El archivo ya existe en {gcs_uri}, no se requiere subida.")


def download_gcs_file(gcs_uri: str, local_dir: Path) -> Path | None:
    """
    Descarga un archivo de GCS a un directorio local.
    """
    # ... (Lógica original de esta función intacta)
    if gcs_path_exists(gcs_uri):
        local_path = local_dir / Path(gcs_uri).name
        logger.info(f"Descargando archivo de GCS: {gcs_uri} -> {local_path}")
        client = _get_gcs_client()
        try:
            bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(local_path))
            return local_path
        except Exception as e:
            logger.error(f"Fallo al descargar archivo desde '{gcs_uri}': {e}", exc_info=True)
            return None
    else:
        logger.warning(f"El archivo no existe en GCS: {gcs_uri}")
        return None


def copy_gcs_directory(source_gcs_dir: str, dest_gcs_dir: str) -> None:
    """
    Copia el contenido de un "directorio" de GCS a otro.
    """
    # ... (Lógica original de esta función intacta)
    client = _get_gcs_client()
    source_bucket_name, source_prefix = source_gcs_dir.replace("gs://", "").split("/", 1)
    dest_bucket_name, dest_prefix = dest_gcs_dir.replace("gs://", "").split("/", 1)
    
    source_bucket = client.bucket(source_bucket_name)
    dest_bucket = client.bucket(dest_bucket_name)
    
    blobs_to_copy = list(source_bucket.list_blobs(prefix=source_prefix.rstrip("/") + "/"))
    logger.info(f"Copiando {len(blobs_to_copy)} objetos de {source_gcs_dir} a {dest_gcs_dir}")
    
    for blob in blobs_to_copy:
        source_blob_name = blob.name
        dest_blob_name = source_blob_name.replace(source_prefix, dest_prefix, 1)
        source_bucket.copy_blob(blob, dest_bucket, dest_blob_name)


def delete_gcs_blob(gcs_uri: str) -> None:
    """
    Elimina un objeto (archivo) específico de GCS.
    """
    # ... (Lógica original de esta función intacta)
    logger.info(f"Eliminando objeto de GCS: {gcs_uri}")
    client = _get_gcs_client()
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
    except exceptions.NotFound:
        logger.warning(f"El objeto a eliminar no se encontró en GCS: {gcs_uri}")
    except Exception as e:
        logger.error(f"Fallo al eliminar objeto en '{gcs_uri}': {e}", exc_info=True)
        raise

def ensure_gcs_path_and_get_local(path_or_uri: str | Path) -> Path:
    # ... (Lógica original de esta función intacta)
    path = Path(path_or_uri)
    if str(path).startswith("gs://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            return download_gcs_file(str(path), Path(tmpdir))
    return path

# --- NUEVA FUNCIÓN AÑADIDA ---
def keep_only_latest_version(base_gcs_prefix: str):
    """
    En un prefijo de GCS, encuentra todos los subdirectorios que parecen
    versionados por timestamp (YYYYMMDDHHMMSS) y borra todos excepto el más reciente.

    Args:
        base_gcs_prefix: La ruta base que contiene los directorios versionados.
                         Ej: "gs://mi-bucket/models/mi-modelo/"
    """
    try:
        client = _get_gcs_client()
        if not base_gcs_prefix.startswith("gs://"):
            raise ValueError("La ruta debe empezar con gs://")
            
        bucket_name, prefix = base_gcs_prefix.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)

        # Asegurarse de que el prefijo termine con /
        if not prefix.endswith('/'):
            prefix += '/'

        # Patrón para encontrar directorios con nombre de timestamp
        timestamp_pattern = re.compile(r"(\d{14})/?$")
        
        blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
        versioned_dirs = []
        # 'page.prefixes' es la forma correcta de listar "subdirectorios"
        for page in blobs.pages:
            for dir_prefix in page.prefixes:
                match = timestamp_pattern.search(dir_prefix)
                if match:
                    versioned_dirs.append(dir_prefix)
        
        if len(versioned_dirs) <= 1:
            logger.info(f"No hay versiones antiguas que limpiar en gs://{bucket_name}/{prefix}")
            return

        # Ordenar de más reciente a más antiguo
        dirs_sorted = sorted(versioned_dirs, reverse=True)
        
        logger.info(f"Se mantendrá la versión más reciente: {dirs_sorted[0]}")
        for old_dir_prefix in dirs_sorted[1:]:
            logger.info(f"🗑️ Eliminando versión anterior y su contenido: {old_dir_prefix}")
            blobs_to_delete = bucket.list_blobs(prefix=old_dir_prefix)
            for blob in blobs_to_delete:
                blob.delete()
                
    except Exception as e:
        logging.warning(f"⚠️ No se pudo realizar la limpieza de versiones antiguas en '{base_gcs_prefix}': {e}")