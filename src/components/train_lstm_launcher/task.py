# src/components/train_lstm_launcher/task.py
"""
Tarea del componente "Lanzador" de Entrenamiento LSTM.

Responsabilidades:
1.  Recibir todos los parámetros necesarios para un entrenamiento.
2.  Configurar un `CustomJob` de Vertex AI.
3.  Pasar los parámetros correctos al script de entrenamiento que se
    ejecutará dentro del CustomJob.
4.  Lanzar el CustomJob y esperar a que complete.
5.  Escribir la ruta del modelo entrenado como un artefacto de salida para KFP.
"""
from __future__ import annotations
import google.cloud.aiplatform as aip

import argparse
import logging
from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform

# --- Configuración del Logging (Sin Cambios) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --- Lógica Principal del Componente (Ajustada) ---
def run_training_job(
    project_id: str,
    region: str,
    pair: str,                      # <-- AJUSTE: Recibe el par
    timeframe: str,                 # <-- AJUSTE: Recibe el timeframe
    params_file: str,
    features_gcs_path: str,
    output_gcs_base_dir: str,
    vertex_training_image_uri: str,
    vertex_machine_type: str,
    vertex_accelerator_type: str,
    vertex_accelerator_count: int,
    vertex_service_account: str,
    trained_lstm_dir_path_output: Path,
) -> None:
    """
    Configura y lanza un CustomJob en Vertex AI para entrenar el modelo LSTM.
    """
    try:
        aip.init(project=project_id, location=region)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        job_display_name = f"train-lstm-{pair.lower()}-{timeframe.lower()}-{timestamp}"
        
        # El directorio de salida ahora incluye el par y timeframe
        output_gcs_dir = f"{output_gcs_base_dir}/{pair}/{timeframe}/{job_display_name}"
        
        logger.info(f"🚀 Lanzando CustomJob de Vertex AI: {job_display_name}")
        logger.info(f"   - Par: {pair}, Timeframe: {timeframe}")
        logger.info(f"   - Directorio de salida: {output_gcs_dir}")
        
        worker_pool_specs = [
            {
                "machine_spec": {
                    "machine_type": vertex_machine_type,
                    "accelerator_type": vertex_accelerator_type,
                    "accelerator_count": vertex_accelerator_count,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": vertex_training_image_uri,
                    "command": ["python", "-m", "src.components.train_lstm.task"],
                    "args": [
                        f"--pair={pair}",
                        f"--timeframe={timeframe}",
                        f"--params-file={params_file}",
                        f"--features-gcs-path={features_gcs_path}",
                        f"--output-gcs-dir={output_gcs_dir}",
                    ],
                },
            }
        ]
        
        job = aip.CustomJob(
            display_name=job_display_name,
            worker_pool_specs=worker_pool_specs,
            stating_dir=str(Path(__file__).parent.parent.parent), # Sube el directorio 'src'
        )
        
        job.run(service_account=vertex_service_account, sync=True)
        
        logger.info(f"✅ Job {job_display_name} completado con éxito.")
        
        # Escribir la ruta de salida para el siguiente componente
        trained_lstm_dir_path_output.parent.mkdir(parents=True, exist_ok=True)
        trained_lstm_dir_path_output.write_text(output_gcs_dir)
        logger.info(f"✍️  Ruta de salida '{output_gcs_dir}' escrita para KFP.")
        
    except Exception as e:
        logger.critical(f"❌ Fallo crítico al lanzar el job de entrenamiento para {pair}: {e}", exc_info=True)
        raise

# --- Punto de Entrada para Ejecución como Script (Ajustado) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanzador de trabajos de entrenamiento LSTM en Vertex AI.")
    
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--pair", required=True) # <-- AJUSTE: Argumento requerido
    parser.add_argument("--timeframe", required=True) # <-- AJUSTE: Argumento requerido
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--features-gcs-path", required=True)
    parser.add_argument("--output-gcs-base-dir", required=True)
    parser.add_argument("--vertex-training-image-uri", required=True)
    parser.add_argument("--vertex-machine-type", required=True)
    parser.add_argument("--vertex-accelerator-type", required=True)
    parser.add_argument("--vertex-accelerator-count", type=int, required=True)
    parser.add_argument("--vertex-service-account", required=True)
    parser.add_argument("--trained-lstm-dir-path-output", type=Path, required=True)
    
    args = parser.parse_args()
    
    run_training_job(
        project_id=args.project_id,
        region=args.region,
        pair=args.pair,
        timeframe=args.timeframe,
        params_file=args.params_file,
        features_gcs_path=args.features_gcs_path,
        output_gcs_base_dir=args.output_gcs_base_dir,
        vertex_training_image_uri=args.vertex_training_image_uri,
        vertex_machine_type=args.vertex_machine_type,
        vertex_accelerator_type=args.vertex_accelerator_type,
        vertex_accelerator_count=args.vertex_accelerator_count,
        vertex_service_account=args.vertex_service_account,
        trained_lstm_dir_path_output=args.trained_lstm_dir_path_output,
    )