# Dockerfile (Versión V5 - Habilitado para GPU con Python 3.10)
# ---------------------------------------------------------------
# 1. Usar una imagen base de TensorFlow con GPU y Python 3.10.
#    NOTA: Esto ajusta la versión de TensorFlow a 2.13.0, que es la
#    última versión oficial con soporte para Python 3.10 en GPU.
FROM tensorflow/tensorflow:2.13.0-gpu

# 2. Mantener las variables de entorno para optimizar la ejecución.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# 3. Directorio de trabajo.
WORKDIR /app

# 4. Copiar y modificar requirements.txt para un entorno GPU.
COPY requirements.txt .

# 5. Ejecutar la instalación en un solo paso robusto.
#    - El primer sed elimina la línea de 'tensorflow' para no reinstalarlo.
#    - El segundo sed elimina el sufijo '+cpu' de torch para instalar la versión GPU.
#    - Finalmente, se instalan los paquetes del archivo ya modificado.
RUN sed -i '/tensorflow/d' requirements.txt && \
    sed -i 's|+cpu||g' requirements.txt && \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copia el código fuente a la imagen.
COPY src/ ./src/

# NO SE NECESITA ENTRYPOINT. La imagen estará lista para recibir cualquier comando.