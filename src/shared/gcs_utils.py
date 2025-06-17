# src/shared/gcs_utils.py
"""
Utilidades para interactuar con Google Cloud Storage (GCS).

Funciones expuestas
-------------------
- get_gcs_client
- upload_gcs_file
- upload_local_directory_to_gcs
- download_gcs_file
- copy_gcs_object
- gcs_path_exists
- delete_gcs_blob
- ensure_gcs_path_and_get_local
- list_gcs_files
- find_latest_gcs_file_in_timestamped_dirs
- keep_only_latest_version
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional

import gcsfs
from google.cloud import storage

# Importar constantes solo si es necesario para la inicialización del cliente.
# from src.shared import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#  CLIENTES Y OPERACIONES BÁSICAS                                             #
# --------------------------------------------------------------------------- #

def get_gcs_client() -> storage.Client:
    """Devuelve un cliente autenticado de GCS."""
    # En un entorno de GCP (como Vertex AI), el cliente se autentica automáticamente.
    return storage.Client()


def upload_local_directory_to_gcs(local_path: Path, gcs_uri: str):
    """Sube el contenido de un directorio local a una URI de GCS."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI de GCS debe empezar por gs://")
    
    bucket_name, gcs_prefix = gcs_uri[5:].split("/", 1)
    bucket = get_gcs_client().bucket(bucket_name)
    
    for local_file in local_path.rglob("*"):
        if local_file.is_file():
            relative_path = local_file.relative_to(local_path)
            blob_name = str(PurePosixPath(gcs_prefix) / relative_path)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_file))
            logger.info("Subido %s → gs://%s/%s", local_file.name, bucket_name, blob_name)


def upload_gcs_file(local_path: Path | str, gcs_uri: str) -> None:
    # ... (lógica existente sin cambios)
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"El archivo local no se encontró: {local_path}")
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI de GCS debe empezar por gs://")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = get_gcs_client().bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(str(local_path))
    logger.info("Subido %s → %s", local_path, gcs_uri)


def download_gcs_file(gcs_uri: str, destination_dir: Path | None = None) -> Path:
    # ... (lógica existente sin cambios)
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI de GCS debe empezar por gs://")
    dest_dir = destination_dir or Path(tempfile.mkdtemp())
    dest_dir.mkdir(parents=True, exist_ok=True)
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    filename = Path(blob_name).name
    local_path = dest_dir / filename
    get_gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))
    logger.info("Descargado %s → %s", gcs_uri, local_path)
    return local_path


def copy_gcs_object(src_uri: str, dst_uri: str) -> None:
    # ... (lógica existente sin cambios)
    if not (src_uri.startswith("gs://") and dst_uri.startswith("gs://")):
        raise ValueError("Ambas URIs deben empezar por gs://")
    client = get_gcs_client()
    src_bucket_name, src_blob_name = src_uri[5:].split("/", 1)
    dst_bucket_name, dst_blob_name = dst_uri[5:].split("/", 1)
    src_bucket = client.bucket(src_bucket_name)
    dst_bucket = client.bucket(dst_bucket_name)
    src_blob = src_bucket.blob(src_blob_name)
    dst_bucket.copy_blob(src_blob, dst_bucket, dst_blob_name)
    logger.info("Copiado %s → %s", src_uri, dst_uri)


def gcs_path_exists(gcs_uri: str) -> bool:
    # ... (lógica existente sin cambios)
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI debe empezar por gs://")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    return get_gcs_client().bucket(bucket_name).blob(blob_name).exists()


def delete_gcs_blob(gcs_uri: str) -> None:
    # ... (lógica existente sin cambios)
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI debe empezar por gs://")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    get_gcs_client().bucket(bucket_name).blob(blob_name).delete()
    logger.info("Eliminado %s", gcs_uri)

# --------------------------------------------------------------------------- #
#  UTILIDADES ADICIONALES                                                     #
# --------------------------------------------------------------------------- #

def ensure_gcs_path_and_get_local(path: str | Path) -> Path:
    # ... (lógica existente sin cambios)
    if str(path).startswith("gs://"):
        return download_gcs_file(str(path))
    return Path(path)


def list_gcs_files(prefix: str, suffix: str | None = None, recursive: bool = True) -> list[str]:
    # ... (lógica existente sin cambios)
    fs = gcsfs.GCSFileSystem()
    no_scheme = prefix.replace("gs://", "").rstrip("/")
    try:
        paths = fs.find(no_scheme, detail=False) if recursive else fs.ls(no_scheme, detail=False)
        if suffix:
            paths = [p for p in paths if p.endswith(suffix)]
        return [f"gs://{p}" for p in paths]
    except FileNotFoundError:
        return []

# --------------------------------------------------------------------------- #
#  BÚSQUEDA Y LIMPIEZA EN DIRECTORIOS VERSIONADOS                             #
# --------------------------------------------------------------------------- #

_TIMESTAMP_RE = re.compile(r"(\d{14})") # YYYYMMDDHHMMSS

def find_latest_gcs_file_in_timestamped_dirs(
    base_gcs_path: str,
    filename: str,
) -> Optional[str]:
    # ... (lógica existente sin cambios)
    if not base_gcs_path.startswith("gs://"):
        raise ValueError("`base_gcs_path` debe comenzar con 'gs://'")
    fs = gcsfs.GCSFileSystem()
    prefix_no_scheme = base_gcs_path.replace("gs://", "").rstrip("/")
    try:
        candidate_files = [p for p in fs.find(prefix_no_scheme, detail=False) if p.endswith("/" + filename)]
    except FileNotFoundError:
        logger.warning("Prefijo GCS inexistente: %s", base_gcs_path)
        return None
    if not candidate_files:
        logger.warning("No se halló %s bajo %s", filename, base_gcs_path)
        return None
    valid = [(PurePosixPath(p).parent.name, p) for p in candidate_files if _TIMESTAMP_RE.fullmatch(PurePosixPath(p).parent.name)]
    if not valid:
        logger.warning("No se hallaron directorios con timestamp válido bajo %s", base_gcs_path)
        return None
    latest_path_no_scheme = sorted(valid, key=lambda t: t[0])[-1][1]
    return f"gs://{latest_path_no_scheme}"

# AJUSTE: Se añade la función de limpieza que faltaba y que causó el error.
def keep_only_latest_version(base_gcs_prefix: str) -> None:
    """
    Dentro de un prefijo GCS, busca subdirectorios cuyo nombre sea un timestamp
    (YYYYMMDDHHMMSS), mantiene solo el más reciente y elimina todos los demás.
    """
    try:
        fs = gcsfs.GCSFileSystem()
        
        # Asegurarse de que el prefijo base existe antes de listar
        if not fs.exists(base_gcs_prefix):
            logger.info("El prefijo base para limpieza no existe: %s. No se hará nada.", base_gcs_prefix)
            return

        # Listar solo los directorios que coincidan con el patrón de timestamp
        dirs = [p for p in fs.ls(base_gcs_prefix, detail=False) if fs.isdir(p) and _TIMESTAMP_RE.search(Path(p).name)]

        if len(dirs) <= 1:
            logger.info("No hay versiones antiguas que limpiar en %s.", base_gcs_prefix)
            return

        # Ordenar por el timestamp para encontrar el más reciente
        dirs.sort(key=lambda p: _TIMESTAMP_RE.search(Path(p).name).group(1), reverse=True)
        
        # Eliminar todos los directorios excepto el primero (el más reciente)
        for old_dir_path in dirs[1:]:
            logger.info("🗑️  Eliminando versión antigua de artefactos: gs://%s", old_dir_path)
            fs.rm(old_dir_path, recursive=True)
            
    except Exception as exc:
        logger.error(
            "Falló la limpieza de versiones antiguas en %s: %s",
            base_gcs_prefix, exc, exc_info=True
        )