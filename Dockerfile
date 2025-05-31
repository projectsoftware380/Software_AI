# Imagen base ligera con Python 3.10
FROM python:3.10-slim

# Evita prompts durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Crea el directorio de trabajo
WORKDIR /app

# Copia archivos del proyecto al contenedor
COPY . /app

# Instala dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias Python del archivo requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Comando por defecto (puedes modificar según tu script principal)
CMD ["python", "train_lstm.py", "--params", "gs://ruta-a-tu-best_params.json"]
