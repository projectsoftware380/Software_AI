# src/deploy/vertex_ai_runner.py

import logging
from datetime import datetime
import google.cloud.aiplatform as aip

logger = logging.getLogger(__name__)

def run_vertex_pipeline(
    pipeline_json_path: str,
    project_id: str,
    region: str,
    pipeline_root: str,
    service_account: str,
    display_name_prefix: str = "algo-trading-pipeline",
    enable_caching: bool = True,
    pipeline_parameters: dict = None,
    sync: bool = True,  # Parámetro para controlar la espera
):
    """
    Lanza una pipeline compilada en Vertex AI y opcionalmente espera a su finalización.

    Devuelve:
        bool: True si la pipeline tiene éxito (o si no es síncrona), False si falla.
    """
    try:
        aip.init(project=project_id, location=region)
        logger.info(f"Cliente de AI Platform inicializado para el proyecto '{project_id}' en la región '{region}'.")

        display_name = f"{display_name_prefix}-{datetime.utcnow():%Y%m%d-%H%M%S}"

        job = aip.PipelineJob(
            display_name=display_name,
            template_path=pipeline_json_path,
            pipeline_root=pipeline_root,
            parameter_values=pipeline_parameters,
            enable_caching=enable_caching,
        )

        logger.info("Configuración del PipelineJob a enviar:")
        logger.info(f"  - Display Name: {display_name}")
        logger.info(f"  - Template Path: {pipeline_json_path}")
        logger.info(f"  - Pipeline Root: {pipeline_root}")
        logger.info(f"  - Service Account: {service_account}")

        logger.info(f"Llamando a job.run(sync={sync}) para lanzar la ejecución...")
        job.run(service_account=service_account, sync=sync)

        logger.info(f"🚀 Pipeline lanzada. Puedes verla en: {job.dashboard_uri}")

        if sync:
            logger.info(f"Pipeline finalizada con estado: {job.state}")
            if job.state == aip.PipelineState.PIPELINE_STATE_SUCCEEDED:
                logger.info("✅ La pipeline de Vertex AI se ha completado exitosamente.")
                return True
            else:
                logger.error(f"❌ La pipeline de Vertex AI ha fallado. Estado final: {job.state}")
                return False
        
        return True # Si no es síncrono, asumimos que el lanzamiento fue exitoso

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al lanzar o monitorizar el pipeline en Vertex AI. Error: {e}")
        return False