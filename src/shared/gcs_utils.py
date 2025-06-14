# src/shared/gcs_utils.py
"""
Utilidades para interactuar con Google Cloud Storage (GCS).

Funciones expuestas
-------------------
- get_gcs_client
- upload_gcs_file
- download_gcs_file
- copy_gcs_object
- gcs_path_exists
- delete_gcs_blob
- ensure_gcs_path_and_get_local
- list_gcs_files
- find_latest_gcs_file_in_timestamped_dirs
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional

import gcsfs
from google.cloud import storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#  CLIENTES Y OPERACIONES BÁSICAS                                             #
# --------------------------------------------------------------------------- #


def get_gcs_client() -> storage.Client:
    """Devuelve un cliente autenticado de GCS."""
    return storage.Client()


def upload_gcs_file(local_path: Path | str, gcs_uri: str) -> None:
    """Sube un archivo local a una URI de GCS."""
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI debe empezar por gs://")

    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = get_gcs_client().bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(str(local_path))
    logger.info("Subido %s → %s", local_path, gcs_uri)


def download_gcs_file(gcs_uri: str, destination_dir: Path | None = None) -> Path:
    """Descarga un objeto de GCS y devuelve la ruta local."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI debe empezar por gs://")

    dest_dir = destination_dir or Path(tempfile.mkdtemp())
    dest_dir.mkdir(parents=True, exist_ok=True)
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    filename = Path(blob_name).name
    local_path = dest_dir / filename

    bucket = get_gcs_client().bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(str(local_path))
    logger.info("Descargado %s → %s", gcs_uri, local_path)
    return local_path


def copy_gcs_object(src_uri: str, dst_uri: str) -> None:
    """Copia un objeto dentro de GCS."""
    if not (src_uri.startswith("gs://") and dst_uri.startswith("gs://")):
        raise ValueError("Las URIs deben empezar por gs://")

    client = get_gcs_client()
    src_bucket, src_blob = src_uri[5:].split("/", 1)
    dst_bucket, dst_blob = dst_uri[5:].split("/", 1)

    src_bucket_ref = client.bucket(src_bucket)
    dst_bucket_ref = client.bucket(dst_bucket)
    dst_bucket_ref.blob(dst_blob).rewrite(src_bucket_ref.blob(src_blob))
    logger.info("Copiado %s → %s", src_uri, dst_uri)


def gcs_path_exists(gcs_uri: str) -> bool:
    """Verifica si un blob existe en GCS."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI debe empezar por gs://")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = get_gcs_client().bucket(bucket_name)
    return bucket.blob(blob_name).exists()


def delete_gcs_blob(gcs_uri: str) -> None:
    """Elimina un blob de GCS."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La URI debe empezar por gs://")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = get_gcs_client().bucket(bucket_name)
    bucket.blob(blob_name).delete()
    logger.info("Eliminado %s", gcs_uri)


# --------------------------------------------------------------------------- #
#  UTILIDADES ADICIONALES                                                     #
# --------------------------------------------------------------------------- #


def ensure_gcs_path_and_get_local(path: str | Path) -> Path:
    """
    Si `path` apunta a GCS, descarga el archivo a un dir temporal y
    devuelve la ruta local; si es local, lo devuelve sin cambios.
    """
    if str(path).startswith("gs://"):
        return download_gcs_file(path)
    return Path(path)


def list_gcs_files(
    prefix: str,
    suffix: str | None = None,
    recursive: bool = True,
) -> list[str]:
    """
    Lista los blobs bajo `prefix`.

    Args:
        prefix: prefijo GCS, con o sin 'gs://'.
        suffix: si se especifica, filtra los blobs que terminen con él.
        recursive: si False, lista solo la primera “carpeta”.

    Returns:
        Rutas completas con esquema 'gs://'.
    """
    fs = gcsfs.GCSFileSystem()
    no_scheme = prefix.replace("gs://", "").rstrip("/")

    paths = (
        fs.find(no_scheme, detail=False) if recursive else fs.ls(no_scheme, detail=False)
    )
    if suffix:
        paths = [p for p in paths if p.endswith(suffix)]

    return [f"gs://{p}" for p in paths]


# --------------------------------------------------------------------------- #
#  BÚSQUEDA EN DIRECTORIOS TIMESTAMPED                                        #
# --------------------------------------------------------------------------- #

_TIMESTAMP_RE = re.compile(r"\d{14}$")  # YYYYMMDDHHMMSS


def find_latest_gcs_file_in_timestamped_dirs(
    base_gcs_path: str,
    filename: str,
) -> Optional[str]:
    """
    Devuelve la ruta `gs://…/<timestamp>/<filename>` más reciente
    encontrada bajo `base_gcs_path`, donde el directorio padre tiene
    formato de timestamp `YYYYMMDDHHMMSS`.

    Args:
        base_gcs_path: Prefijo como
            ``gs://bucket/params/LSTM_v3/EURUSD``.
        filename: Archivo a localizar, ej. ``best_params.json``.

    Returns:
        Ruta completa con esquema `gs://` o `None` si no se halla.
    """
    if not base_gcs_path.startswith("gs://"):
        raise ValueError("`base_gcs_path` debe comenzar con 'gs://'")

    fs = gcsfs.GCSFileSystem()
    prefix_no_scheme = base_gcs_path.replace("gs://", "").rstrip("/")

    # Busca recursivamente todos los blobs que terminen en filename
    try:
        candidate_files = [
            p for p in fs.find(prefix_no_scheme, detail=False)
            if p.endswith("/" + filename)
        ]
    except FileNotFoundError:
        logger.warning("Prefijo GCS inexistente: %s", base_gcs_path)
        return None

    if not candidate_files:
        logger.warning("No se halló %s bajo %s", filename, base_gcs_path)
        return None

    def extract_ts(path: str) -> str:
        """Nombre del directorio padre (timestamp)."""
        return PurePosixPath(path).parent.name

    # Filtra aquellos cuyo directorio padre cumpla con YYYYMMDDHHMMSS
    valid = [
        (extract_ts(p), p) for p in candidate_files
        if _TIMESTAMP_RE.fullmatch(extract_ts(p))
    ]
    if not valid:
        logger.warning(
            "No se hallaron directorios con timestamp válido bajo %s", base_gcs_path
        )
        return None

    # Orden alfabético equivale a orden cronológico → último es el más reciente
    latest_path_no_scheme = sorted(valid, key=lambda t: t[0])[-1][1]
    return f"gs://{latest_path_no_scheme}"
