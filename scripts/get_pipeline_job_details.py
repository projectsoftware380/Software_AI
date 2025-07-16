# scripts/get_pipeline_job_details.py

import argparse
import logging
import google.cloud.aiplatform as aip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_job_details(job_id: str, project_id: str, region: str):
    """Obtiene y muestra los detalles de un PipelineJob de Vertex AI, incluyendo detalles de tareas fallidas."""
    try:
        aip.init(project=project_id, location=region)
        job = aip.PipelineJob.get(job_id) 
        
        logger.info(f"Estado del PipelineJob '{job_id}': {job.state}")
        if job.state == aip.PipelineState.PIPELINE_STATE_FAILED:
            logger.error(f"Mensaje de error general del PipelineJob: {job.error}")
            
            logger.info("Detalles de las tareas:")
            for task_id, task_detail in job.task_details.items():
                logger.info(f"  Tarea: {task_id}, Estado: {task_detail.state}")
                if task_detail.state == aip.PipelineState.PIPELINE_STATE_FAILED:
                    logger.error(f"    Mensaje de error de la tarea '{task_id}': {task_detail.error}")
                    if task_detail.logs_uri:
                        logger.info(f"    Logs de la tarea '{task_id}': {task_detail.logs_uri}")

    except Exception as e:
        logger.error(f"Error al obtener detalles del PipelineJob '{job_id}': {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtener detalles de un PipelineJob de Vertex AI.")
    parser.add_argument("--job-id", required=True, help="ID del PipelineJob.")
    parser.add_argument("--project-id", required=True, help="ID del proyecto de Google Cloud.")
    parser.add_argument("--region", default="europe-west1", help="Regin del PipelineJob.")
    args = parser.parse_args()

    get_job_details(args.job_id, args.project_id, args.region)
