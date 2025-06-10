# src/pipeline/main.py
"""
Define y compila la pipeline de KFP v3 para el entrenamiento de modelos de trading.

Este script es el orquestador principal que carga componentes desde archivos .yaml externos.
Está diseñado para aceptar la URI de la imagen Docker como un parámetro de línea de comandos,
permitiendo ejecuciones dinámicas y reproducibles sin modificar el código fuente.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

# KFP y Vertex AI Imports
from kfp import compiler
from kfp.dsl import pipeline
from kfp.components import load_component_from_file
import google.cloud.aiplatform as aip

# Cargar constantes y configuraciones del proyecto
from src.shared import constants

# --- Carga de Componentes ---
COMPONENTS_DIR = Path(__file__).parent.parent / "components"

component_op_factory = {
    "data_ingestion": load_component_from_file(
        COMPONENTS_DIR / "data_ingestion/component.yaml"
    ),
    "data_preparation": load_component_from_file(
        COMPONENTS_DIR / "data_preparation/component.yaml"
    ),
    "hyperparam_tuning": load_component_from_file(
        COMPONENTS_DIR / "hyperparam_tuning/component.yaml"
    ),
    "train_lstm_launcher": load_component_from_file(
        COMPONENTS_DIR / "train_lstm_launcher/component.yaml"
    ),
    "prepare_rl_data": load_component_from_file(
        COMPONENTS_DIR / "prepare_rl_data/component.yaml"
    ),
    "train_rl": load_component_from_file(
        COMPONENTS_DIR / "train_rl/component.yaml"
    ),
    "backtest": load_component_from_file(
        COMPONENTS_DIR / "backtest/component.yaml"
    ),
    "model_promotion": load_component_from_file(
        COMPONENTS_DIR / "model_promotion/component.yaml"
    ),
}


# --- Definición de la Pipeline ---
@pipeline(
    name="algo-trading-mlops-modular-pipeline-v3",
    description="KFP v3 Pipeline modular para entrenar y desplegar modelos de trading algorítmico.",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline(
    pair: str = constants.DEFAULT_PAIR,
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials: int = constants.DEFAULT_N_TRIALS,
    backtest_features_gcs_path: str = f"{constants.DATA_PATH}/{constants.DEFAULT_PAIR}/{constants.DEFAULT_TIMEFRAME}/{constants.DEFAULT_PAIR}_{constants.DEFAULT_TIMEFRAME}_unseen.parquet",
    vertex_machine_type: str = constants.DEFAULT_VERTEX_LSTM_MACHINE_TYPE,
    vertex_accelerator_type: str = constants.DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE,
    vertex_accelerator_count: int = constants.DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT,
    # === AJUSTE CORREGIDO: Se declara el parámetro sin valor por defecto ===
    common_image_uri: str
    # =======================================================================
):
    # 1. Ingestión de Datos
    ingest_task = component_op_factory["data_ingestion"](
        pair=pair,
        timeframe=timeframe,
        project_id=constants.PROJECT_ID,
        polygon_secret_name=constants.POLYGON_API_KEY_SECRET_NAME,
        start_date="2010-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        min_rows=100000,
    )
    # Forzar la imagen para este componente. Es una buena práctica si no estás
    # seguro de que el YAML del componente tenga la imagen correcta.
    ingest_task.container.set_image(common_image_uri)

    # 2. Preparación de Datos para Optimización
    prepare_opt_data_task = component_op_factory["data_preparation"](
        pair=pair,
        timeframe=timeframe,
    ).after(ingest_task)
    prepare_opt_data_task.container.set_image(common_image_uri)


    # 3. Optimización de Hiperparámetros
    tuning_task = component_op_factory["hyperparam_tuning"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"],
        pair=pair,
        timeframe=timeframe,
        n_trials=n_trials,
    )
    tuning_task.container.set_image(common_image_uri)

    # 4. Lanzar Job de Entrenamiento del LSTM
    train_lstm_task = component_op_factory["train_lstm_launcher"](
        project_id=constants.PROJECT_ID,
        region=constants.REGION,
        pair=pair,
        timeframe=timeframe,
        params_path=tuning_task.outputs["best_params_path"],
        output_gcs_base_dir=constants.LSTM_MODELS_PATH,
        vertex_training_image_uri=common_image_uri, # Se pasa el parámetro dinámico
        vertex_machine_type=vertex_machine_type,
        vertex_accelerator_type=vertex_accelerator_type,
        vertex_accelerator_count=vertex_accelerator_count,
        vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
    )
    # Este componente lanza otro job, así que no se le establece la imagen a él,
    # sino que se le pasa la URI para que él la use.

    # 5. Preparar Datos para el Agente RL
    prepare_rl_data_task = component_op_factory["prepare_rl_data"](
        lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        pair=pair,
        timeframe=timeframe,
        output_gcs_base_dir=constants.RL_DATA_INPUTS_PATH,
    )
    prepare_rl_data_task.container.set_image(common_image_uri)

    # 6. Entrenar Agente RL (PPO)
    train_rl_task = component_op_factory["train_rl"](
        params_path=f"{train_lstm_task.outputs['trained_lstm_dir_path']}/params.json",
        rl_data_path=prepare_rl_data_task.outputs["rl_data_path"],
        pair=pair,
        timeframe=timeframe,
        output_gcs_base_dir=constants.RL_MODELS_PATH,
        tensorboard_logs_base_dir=constants.TENSORBOARD_LOGS_PATH,
    )
    train_rl_task.container.set_image(common_image_uri)

    # 7. Ejecutar Backtest
    backtest_task = component_op_factory["backtest"](
        lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
        features_path=backtest_features_gcs_path,
        pair=pair,
        timeframe=timeframe,
    )
    backtest_task.container.set_image(common_image_uri)

    # 8. Decidir Promoción a Producción
    promotion_task = component_op_factory["model_promotion"](
        new_metrics_dir=backtest_task.outputs["output_gcs_dir"],
        new_lstm_artifacts_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        new_rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
        pair=pair,
        timeframe=timeframe,
        production_base_dir=constants.PRODUCTION_MODELS_PATH,
    )
    promotion_task.container.set_image(common_image_uri)

# --- Compilación y Ejecución de la Pipeline ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common-image-uri",
        type=str,
        required=True,
        help="La URI completa de la imagen Docker a usar en los componentes."
    )
    args = parser.parse_args()

    pipeline_filename = "algo_trading_mlops_modular_pipeline_v3.json"
    
    compiler.Compiler().compile(
        pipeline_func=trading_pipeline,
        package_path=pipeline_filename
    )
    print(f"✅ Pipeline compilada a {pipeline_filename}")

    SUBMIT_TO_VERTEX = os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() == "true"

    if SUBMIT_TO_VERTEX:
        print("\n🚀 Iniciando sumisión y ejecución de la pipeline en Vertex AI...")
        aip.init(project=constants.PROJECT_ID, location=constants.REGION)
        
        job_display_name = f"algo-trading-v3-{constants.DEFAULT_PAIR}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        job = aip.PipelineJob(
            display_name=job_display_name,
            template_path=pipeline_filename,
            pipeline_root=constants.PIPELINE_ROOT,
            enable_caching=False,
            parameter_values={
                "pair": constants.DEFAULT_PAIR,
                "timeframe": constants.DEFAULT_TIMEFRAME,
                "n_trials": constants.DEFAULT_N_TRIALS,
                "common_image_uri": args.common_image_uri
            }
        )
        
        print(f"Enviando PipelineJob '{job_display_name}' con la imagen '{args.common_image_uri}'...")
        job.run()
        print(f"✅ PipelineJob enviado. Puedes verlo en la consola de Vertex AI.")