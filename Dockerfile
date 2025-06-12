# Dockerfile (Versión V4 - Habilitado para GPU)
# --------------------------------------------
# 1. Usar una imagen base de TensorFlow con GPU y la misma versión de Python (3.10)
# Esta imagen ya contiene los drivers de NVIDIA (CUDA) y las librerías necesarias.
# La etiqueta '2.16.1-gpu' corresponde a la versión de TensorFlow en tu requirements.txt
FROM tensorflow/tensorflow:2.16.1-gpu

# 2. Mantener las variables de entorno para optimizar la ejecución.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# 3. Directorio de trabajo.
WORKDIR /app

# 4. Copiar y modificar requirements.txt para un entorno GPU.
# Se copia el archivo original y luego se modifica DENTRO del contenedor.
COPY requirements.txt .
# - Se elimina la línea de 'tensorflow' para no reinstalarlo (ya viene en la imagen base).
# - Se elimina el sufijo '+cpu' de torch para que pip instale la versión con soporte CUDA.
RUN sed -i '/tensorflow/d' requirements.txt && \
    sed -i 's|+cpu||g' requirements.txt

# 5. Instalación de dependencias del archivo modificado.
# Se elimina --extra-index-url para que pip resuelva las dependencias GPU automáticamente.
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copia el código fuente a la imagen.
COPY src/ ./src/

# NO SE NECESITA ENTRYPOINT. La imagen estará lista para recibir cualquier comando.