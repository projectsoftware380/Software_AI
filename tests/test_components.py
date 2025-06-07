# tests/test_components.py
"""
Pruebas unitarias y de integración para los componentes de la pipeline.

Este archivo utiliza pytest y unittest.mock para probar la lógica de cada
script `task.py` de forma aislada. Las dependencias externas (como las
llamadas a APIs de GCP o a servicios de datos) se "mockean" para que las
pruebas sean rápidas, determinísticas y no dependan de credenciales o
recursos en la nube.
"""

import subprocess
import json
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest

# --- Constantes para Pruebas ---
TEST_PAIR = "EURUSD"
TEST_TIMEFRAME = "15minute"
TEST_PROJECT_ID = "test-project"
TEST_GCS_BUCKET = "test-bucket"

# --- Pruebas para el Componente: data_ingestion ---

# Se usan "patches" para interceptar y simular las llamadas a funciones externas.
@patch('src.components.data_ingestion.task.pubsub_v1.PublisherClient')
@patch('src.components.data_ingestion.task.requests.Session')
@patch('src.components.data_ingestion.task.get_polygon_api_key')
@patch('src.shared.gcs_utils.upload_gcs_file')
@patch('src.shared.gcs_utils.delete_gcs_blob')
@patch('src.shared.gcs_utils.gcs_path_exists')
def test_data_ingestion_task_success(
    mock_path_exists, mock_delete_blob, mock_upload, mock_get_api_key, mock_session, mock_pubsub
):
    """
    Prueba el flujo de éxito del script de ingestión de datos.
    Verifica que se llamen a las funciones correctas con los argumentos esperados.
    """
    # 1. Configurar los mocks
    mock_path_exists.return_value = True  # Simula que un archivo antiguo existe
    mock_get_api_key.return_value = "fake_polygon_api_key"

    # Simular una respuesta exitosa de la API de Polygon
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"t": 123456, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.05, "v": 100}] * 10
    }
    mock_session.return_value.get.return_value = mock_response

    # 2. Ejecutar el script como un subproceso
    cmd = [
        "python", "-m", "src.components.data_ingestion.task",
        "--pair", TEST_PAIR,
        "--timeframe", TEST_TIMEFRAME,
        "--project-id", TEST_PROJECT_ID,
        "--min-rows", "5", # Un valor bajo para la prueba
        f"--gcs-data-path=gs://{TEST_GCS_BUCKET}/data"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # 3. Realizar aserciones (verificaciones)
    assert result.returncode == 0, f"El script falló con el error: {result.stderr}"
    
    # Verificar que las funciones mockeadas fueron llamadas
    mock_path_exists.assert_called_once()
    mock_delete_blob.assert_called_once()
    mock_get_api_key.assert_called_once()
    mock_upload.assert_called_once()
    mock_pubsub.return_value.publish.assert_called_once()
    
    # Verificar que la subida a GCS se intentó con la ruta correcta
    final_gcs_path = mock_upload.call_args[0][1]
    assert f"gs://{TEST_GCS_BUCKET}/data/{TEST_PAIR}/{TEST_TIMEFRAME}" in final_gcs_path


# --- Pruebas para el Componente: data_preparation ---

@patch('src.shared.gcs_utils.upload_gcs_file')
@patch('src.shared.gcs_utils.download_gcs_file')
def test_data_preparation_task(mock_download, mock_upload, tmp_path):
    """
    Prueba el flujo de éxito del script de preparación de datos.
    Simula la descarga de un parquet y verifica que se suba un resultado.
    """
    # 1. Crear datos de prueba y guardarlos en un archivo parquet local temporal
    dummy_data = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range(start='2020-01-01', periods=100, freq='h')),
        'open': np.random.rand(100), 'high': np.random.rand(100),
        'low': np.random.rand(100), 'close': np.random.rand(100)
    })
    local_parquet = tmp_path / "input.parquet"
    dummy_data.to_parquet(local_parquet)
    
    # Configurar el mock para que "descargue" nuestro archivo local
    mock_download.return_value = local_parquet

    # 2. Ejecutar el script
    cmd = [
        "python", "-m", "src.components.data_preparation.task",
        "--pair", TEST_PAIR,
        "--timeframe", TEST_TIMEFRAME,
        "--years-to-keep", "1"
    ]
    # El `task.py` construye la ruta de entrada, por lo que el mock de descarga la interceptará.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    # 3. Aserciones
    assert result.returncode == 0, f"El script falló con el error: {result.stderr}"
    mock_download.assert_called_once()
    mock_upload.assert_called_once()
    
    # Verificar que la salida del script (la ruta del archivo creado) es correcta
    output_path = result.stdout.strip()
    assert "recent.parquet" in output_path
    assert f"{TEST_PAIR}_{TEST_TIMEFRAME}" in output_path


# --- Pruebas para Componentes de ML (ej. model_promotion) ---

@patch('src.shared.gcs_utils.copy_gcs_object')
@patch('src.shared.gcs_utils.upload_gcs_file')
@patch('src.components.model_promotion.task._load_metrics')
def test_model_promotion_task_promotes(mock_load_metrics, mock_upload, mock_copy):
    """Prueba el caso en que el modelo SÍ es promovido."""
    # 1. Configurar mocks para simular métricas excelentes
    mock_load_metrics.side_effect = [
        { # new_metrics
            "trades": 200, "sharpe": 2.5, "profit_factor": 2.0,
            "win_rate": 0.6, "max_drawdown": -5.0, "expectancy": 0.1
        },
        { # prod_metrics
            "trades": 150, "sharpe": 1.0, "profit_factor": 1.5,
            "win_rate": 0.5, "max_drawdown": -10.0, "expectancy": 0.05
        }
    ]

    # 2. Ejecutar script
    cmd = [
        "python", "-m", "src.components.model_promotion.task",
        "--new-metrics-dir", "gs://fake/metrics/dir",
        "--new-lstm-artifacts-dir", "gs://fake/lstm/dir",
        "--new-rl-model-path", "gs://fake/rl/model.zip",
        "--pair", TEST_PAIR,
        "--timeframe", TEST_TIMEFRAME,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    # 3. Aserciones
    assert result.returncode == 0, f"El script falló con el error: {result.stderr}"
    # Se debe haber llamado a la copia de artefactos
    assert mock_copy.call_count > 0
    assert mock_upload.call_count == 0 # En la versión actual se usa copy, no upload
    # El script debe imprimir "true"
    assert result.stdout.strip() == "true"

# Nota: Un conjunto de pruebas completo también incluiría casos de fallo,
# como datos de entrada inválidos, respuestas de API con errores,
# y casos donde el modelo no es promovido. Este archivo sirve como una
# base sólida y una demostración del enfoque de prueba.