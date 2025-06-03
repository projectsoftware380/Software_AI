#!/bin/bash

# Salir inmediatamente si un comando falla
set -e

TARGET_DIR="/tmp/code_execution" # Usar un subdirectorio en /tmp para mayor limpieza

echo "Creando directorio de destino en ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}" # Cambiar al directorio de trabajo

echo "Descargando scripts desde GCS a ${TARGET_DIR}..."
# El flag -r es necesario si 'code' contiene subdirectorios, pero para *.py directamente en code/, no es estrictamente necesario.
# Si solo son archivos .py en la raíz de 'code/', el siguiente comando es más simple:
gsutil -m cp "gs://trading-ai-models-460823/code/*.py" .
# Si 'code' tuviera estructura de directorios que quieres replicar, usarías:
# gsutil -m cp -r "gs://trading-ai-models-460823/code/*" .

# Verificar si se descargaron los archivos (opcional pero recomendado)
if ! ls -1qA . | grep -q "."; then # Verifica si el directorio está vacío
    echo "Error: No se descargaron archivos desde GCS o la carpeta de código está vacía."
    exit 1
fi

echo "Archivos descargados en ${TARGET_DIR}:"
ls -la

# Pasar todos los argumentos recibidos por run.sh al script de Python
# Esto permite que KFP/Vertex AI pase argumentos al script de Python a través del ENTRYPOINT.
# Por ejemplo, si tu train_lstm.py acepta --params, --pair, etc.
SCRIPT_TO_RUN="train_lstm.py" # Define el script principal aquí

if [ -f "${SCRIPT_TO_RUN}" ]; then
    echo "Ejecutando el script: python ${SCRIPT_TO_RUN} "$@""
    python "${SCRIPT_TO_RUN}" "$@"
else
    echo "Error: El script principal ${SCRIPT_TO_RUN} no se encontró después de la descarga."
    exit 1
fi

echo "Ejecución del script finalizada."