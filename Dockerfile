# Paso 1: Usar la imagen OFICIAL de TensorFlow para la versión 2.15.0 con GPU.
# Esta imagen es el estándar de la industria, mantenida por el equipo de TensorFlow.
# Contiene Python 3.10 y todas las librerías CUDA/cuDNN necesarias pre-configuradas.
FROM tensorflow/tensorflow:2.15.0-gpu

# Paso 2: Instalar las dependencias de sistema que tu proyecto necesita.
# Mantenemos la instalación de cairo que ya tenías.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libcairo2-dev && \
    # Limpiar el caché para mantener la imagen ligera
    rm -rf /var/lib/apt/lists/*

# Paso 3: Establecer el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Paso 4: Copiar el archivo de requerimientos de Python.
COPY requirements.txt .

# Paso 5: Instalar las dependencias de Python.
# Como la imagen base ya incluye TensorFlow, lo eliminamos del requirements.txt
# usando 'sed' antes de instalar el resto de las librerías.
RUN sed -i '/tensorflow/d' requirements.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Paso 6: Copiar todo el código de tu proyecto al contenedor.
COPY . .

# Paso 7: Definir un comando por defecto (será sobreescrito por Kubeflow).
CMD ["/bin/bash"]