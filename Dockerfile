# Imagen base ligera con Python 3.10
FROM python:3.10-slim

# Configura variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Crea un usuario no-root para la aplicación
ARG APP_USER=appuser
ARG APP_GROUP=appgroup
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${APP_GROUP} && \
    useradd -u ${UID} -g ${APP_GROUP} -ms /bin/bash ${APP_USER}

# Crea el directorio de trabajo y establece permisos
WORKDIR /app
COPY --chown=${APP_USER}:${APP_GROUP} requirements.txt .

# Instala dependencias del sistema necesarias como root
# y luego limpia la caché de apt para reducir el tamaño de la imagen
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias Python (aún como root para este paso,
# pero se podría hacer después de cambiar a APP_USER si se instala en el home del usuario)
# o si se usa un virtual environment propiedad de APP_USER.
# Por simplicidad para este ejemplo, instalamos globalmente y luego cambiamos de usuario.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copia el resto del código de la aplicación y establece permisos
COPY --chown=${APP_USER}:${APP_GROUP} . .

# Cambia al usuario no-root
USER ${APP_USER}

# Comando por defecto (ajusta los argumentos según sea necesario o pásalos en tiempo de ejecución)
# Es mejor no tener un CMD por defecto tan específico si la imagen se usará para múltiples scripts
# o si los parámetros van a cambiar. Considera esto como un placeholder.
# CMD ["python", "train_lstm.py", "--params", "gs://ruta-a-tu-best_params.json"]
# Un CMD más genérico podría ser:
# CMD ["python"]
# O dejarlo para que se especifique en la definición del componente KFP o del Job de Vertex AI.
# Por ahora, lo comentaré para mayor flexibilidad.

# Si necesitas un entrypoint o un CMD específico, descomenta y ajusta:
# Por ejemplo, para que un componente KFP simplemente tenga el código y Python disponible:
# ENTRYPOINT ["python"]