# src/shared/gcs_utils.py
"""
Utilidades para interactuar con Google Cloud Storage (GCS).
Incluye operaciones de subida, descarga, copia y limpieza de artefactos.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional

import gcsfs
from google.cloud import storage

# Se importa `constants` para poder usar el PROJECT_ID en gcsfs.
from src.shared import constants

logger = logging.getLogger(__name__)
# Asegurarse de que el logger esté configurado si aún no lo está.
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Descompone una URI de GCS en ``(bucket, blob)``.

    Valida que la URI comience con ``gs://`` y que incluya tanto el nombre del
    bucket como la ruta del objeto.
    """
    if not uri.startswith("gs://"):
        raise ValueError(
            f"La URI de GCS debe empezar por gs://, pero se recibió: {uri}"
        )

    without_scheme = uri[5:]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"URI de GCS inválida: {uri}")

    return parts[0], parts[1]


def get_gcs_client() -> storage.Client:
    """Devuelve un cliente autenticado de GCS."""
    return storage.Client()


def upload_local_directory_to_gcs(local_path: Path, gcs_uri: str):
    """Sube un directorio local completo a una ruta de GCS."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"La URI de GCS debe empezar por gs://, pero se recibió: {gcs_uri}")
    
    bucket_name, gcs_prefix = gcs_uri[5:].split("/", 1)
    bucket = get_gcs_client().bucket(bucket_name)
    
    for local_file in local_path.rglob("*"):
        if local_file.is_file():
            relative_path = local_file.relative_to(local_path)
            # Asegura la compatibilidad de rutas entre Windows y POSIX
            blob_name = str(PurePosixPath(gcs_prefix) / relative_path)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_file))
            logger.info("Subido %s → gs://%s/%s", local_file.name, bucket_name, blob_name)


def upload_gcs_file(local_path: Path | str, gcs_uri: str) -> None:
    """Sube un archivo local a una URI de GCS."""
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"El archivo local no se encontró: {local_path}")
    
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"La URI de GCS debe empezar por gs://, pero se recibió: {gcs_uri}")
    
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = get_gcs_client().bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(str(local_path))
    logger.info("Subido %s → %s", local_path, gcs_uri)


def download_gcs_file(gcs_uri: str, destination_dir: Path | None = None) -> Path:
    """Descarga un objeto de GCS y devuelve la ruta local."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"La URI de GCS debe empezar por gs://, pero se recibió: {gcs_uri}")
        
    dest_dir = destination_dir or Path(tempfile.mkdtemp())
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    local_path = dest_dir / Path(blob_name).name
    
    # Use the ``Path`` object directly so callers and tests can assert against
    # the same type. ``google-cloud-storage`` accepts both ``str`` and ``Path``
    # instances.
    get_gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    logger.info("Descargado %s → %s", gcs_uri, local_path)
    return local_path


def copy_gcs_object(src_uri: str, dst_uri: str) -> None:
    """Copia un objeto dentro de GCS."""
    if not (src_uri.startswith("gs://") and dst_uri.startswith("gs://")):
        raise ValueError("Ambas URIs deben empezar por gs://")
        
    client = get_gcs_client()
    src_bucket_name, src_blob_name = src_uri[5:].split("/", 1)
    dst_bucket_name, dst_blob_name = dst_uri[5:].split("/", 1)
    
    src_bucket = client.bucket(src_bucket_name)
    dst_bucket = client.bucket(dst_bucket_name)
    src_blob = src_bucket.blob(src_blob_name)
    
    dst_bucket.blob(dst_blob_name).rewrite(src_blob)
    logger.info("Copiado %s → %s", src_uri, dst_uri)


def gcs_path_exists(gcs_uri: str) -> bool:
    """Verifica si un blob o prefijo existe en GCS."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"La URI de GCS debe empezar por gs://, pero se recibió: {gcs_uri}")
        
    fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
    return fs.exists(gcs_uri)


def delete_gcs_blob(gcs_uri: str) -> None:
    """Elimina un blob de GCS."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"La URI de GCS debe empezar por gs://, pero se recibió: {gcs_uri}")
        
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    get_gcs_client().bucket(bucket_name).blob(blob_name).delete()
    logger.info("Eliminado %s", gcs_uri)


def ensure_gcs_path_and_get_local(path: str | Path) -> Path:
    """Si `path` es GCS, lo descarga y devuelve la ruta local; si no, solo lo convierte a Path."""
    if str(path).startswith("gs://"):
        return download_gcs_file(str(path))
    return Path(path)


def list_gcs_files(prefix: str, suffix: str | None = None, recursive: bool = True) -> list[str]:
    """Lista blobs bajo un prefijo, devolviendo rutas `gs://`."""
    fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
    prefix_no_scheme = prefix.replace("gs://", "").rstrip("/")
    try:
        paths = fs.find(prefix_no_scheme, detail=False) if recursive else fs.ls(prefix_no_scheme, detail=False)
        if suffix:
            paths = [p for p in paths if p.endswith(suffix)]
        return [f"gs://{p}" for p in paths]
    except FileNotFoundError:
        return []

# --------------------------------------------------------------------------- #
#  BÚSQUEDA Y LIMPIEZA EN DIRECTORIOS VERSIONADOS                             #
# --------------------------------------------------------------------------- #

_TIMESTAMP_RE = re.compile(r"(\d{14})") # Formato YYYYMMDDHHMMSS

def find_latest_gcs_file_in_timestamped_dirs(
    base_gcs_path: str,
    filename: str,
) -> Optional[str]:
    """
    Encuentra la ruta `gs://.../<timestamp>/<filename>` más reciente bajo `base_gcs_path`.
    """
    if not base_gcs_path.startswith("gs://"):
        raise ValueError(f"`base_gcs_path` debe comenzar con 'gs://', pero se recibió: {base_gcs_path}")

    all_files = list_gcs_files(base_gcs_path, suffix=f"/{filename}", recursive=True)
    if not all_files:
        logger.warning("No se halló el archivo '%s' bajo %s", filename, base_gcs_path)
        return None

    valid = []
    for file_path in all_files:
        parent_dir_name = Path(file_path).parent.name
        if _TIMESTAMP_RE.fullmatch(parent_dir_name):
            valid.append((parent_dir_name, file_path))

    if not valid:
        logger.warning("No se hallaron directorios con formato de timestamp válido bajo %s", base_gcs_path)
        return None

    latest_path = sorted(valid, key=lambda t: t[0])[-1][1]
    return latest_path

def keep_only_latest_version(base_gcs_prefix: str) -> None:
    """
    Dentro de un prefijo GCS, busca subdirectorios con nombre de timestamp,
    mantiene solo el más reciente y elimina todos los demás.
    """
    try:
        # --- INICIO DE LA CORRECCIÓN ---
        # Se añade una lógica para reparar automáticamente una URI mal formada (gs:/ -> gs://)
        if base_gcs_prefix.startswith("gs:/") and not base_gcs_prefix.startswith("gs://"):
            corrected_prefix = base_gcs_prefix.replace("gs:/", "gs://", 1)
            logger.warning(
                "Se recibió una URI de GCS mal formada. Corrigiendo '%s' a '%s'",
                base_gcs_prefix,
                corrected_prefix,
            )
            base_gcs_prefix = corrected_prefix
        # --- FIN DE LA CORRECCIÓN ---
            
        if not base_gcs_prefix.startswith("gs://"):
             raise ValueError(f"El prefijo GCS para limpieza debe ser una URI válida (gs://...), pero se recibió: {base_gcs_prefix}")
        
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
        
        prefix_no_scheme = base_gcs_prefix.replace("gs://", "")

        if not fs.exists(prefix_no_scheme):
            logger.info("El prefijo base para limpieza no existe: %s. No se hará nada.", base_gcs_prefix)
            return

        all_dirs = fs.ls(prefix_no_scheme, detail=False)
        timestamped_dirs = [p for p in all_dirs if fs.isdir(p) and _TIMESTAMP_RE.search(Path(p).name)]

        if len(timestamped_dirs) <= 1:
            logger.info("No hay versiones antiguas que limpiar en %s.", base_gcs_prefix)
            return

        timestamped_dirs.sort(key=lambda p: _TIMESTAMP_RE.search(Path(p).name).group(1), reverse=True)
        
        for old_dir_path in timestamped_dirs[1:]:
            logger.info("🗑️ Eliminando versión antigua de artefactos: gs://%s", old_dir_path)
            fs.rm(old_dir_path, recursive=True)
            
    except Exception as exc:
        logger.error(
            "Falló la limpieza de versiones antiguas en %s: %s",
            base_gcs_prefix, exc, exc_info=True
        )