# Dockerfile (Versión V3 Simplificada)
# -----------
# Imagen base de Python.
FROM python:3.10-slim-bookworm

# Variables de entorno para optimizar la ejecución.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Directorio de trabajo.
WORKDIR /app

# Instalación de dependencias (usa el caché de Docker si requirements.txt no ha cambiado).
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copia el código fuente a la imagen. Este es el paso clave.
COPY src/ ./src/

# NO SE NECESITA ENTRYPOINT. La imagen estará lista para recibir cualquier comando.