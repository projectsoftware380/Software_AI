# Imagen base ligera con Python 3.10
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Crea el directorio de trabajo
WORKDIR /app

# Copia primero solo requirements.txt para cachear pip install
COPY requirements.txt .

# Instala dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias Python
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 🔁 Ahora copia el resto del código
COPY . .

# Comando por defecto (puede ajustarse según tu uso)
CMD ["python", "train_lstm.py", "--params", "gs://ruta-a-tu-best_params.json"]
