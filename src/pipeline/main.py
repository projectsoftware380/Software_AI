# src/pipeline/main.py
import argparse
import os
from datetime import datetime
from pathlib import Path

from kfp import compiler
from kfp.dsl import pipeline
from kfp.components import load_component_from_file
import google.cloud.aiplatform as aip

from src.shared import constants

# --- Argument Parser ---
# Se mueve aquí para que esté disponible globalmente en el script
parser = argparse.ArgumentParser()
parser.add_argument(
    "--common-image-uri",
    type=str,
    required=True,
    help="La URI completa de la imagen Docker a usar en los componentes."
)
args = parser.parse_args()


# --- Carga de Componentes ---
COMPONENTS_DIR = Path(__file__).parent.parent / "components"

component_op_factory = {
    "data_ingestion": load_component_from_file(COMPONENTS_DIR / "data_ingestion/component.yaml"),
    "data_preparation": load_component_from_file(COMPONENTS_DIR / "data_preparation/component.yaml"),
    "hyperparam_tuning": load_component_from_file(COMPONENTS_DIR / "hyperparam_tuning/component.yaml"),
    "train_lstm_launcher": load_component_from_file(COMPONENTS_DIR / "train_lstm_launcher/component.yaml"),
    "prepare_rl_data": load_component_from_file(COMPONENTS_DIR / "prepare_rl_data/component.yaml"),
    "train_rl": load_component_from_file(COMPONENTS_DIR / "train_rl/component.yaml"),
    "backtest": load_component_from_file(COMPONENTS_DIR / "backtest/component.yaml"),
    "model_promotion": load_component_from_file(COMPONENTS_DIR / "model_promotion/component.yaml"),
}

# === AJUSTE CLAVE: Sobreescribir la imagen de cada componente cargado ===
# Iteramos sobre todos los componentes que hemos cargado desde YAML.
for name, component_op in component_op_factory.items():
    # El lanzador es especial: su imagen es la del lanzador, no la del entrenamiento.
    # El resto de componentes sí usan la misma imagen común.
    if name != "train_lstm_launcher":
        component_op.component_spec.implementation.container.image = args.common_image_uri
# =======================================================================


# --- Definición de la Pipeline ---
@pipeline(
    name="algo-trading-mlops-modular-pipeline-v3",
    description="KFP v3 Pipeline modular para entrenar y desplegar modelos de trading algorítmico.",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline(
    # La pipeline ya no necesita recibir 'common_image_uri' porque los componentes ya están modificados
    pair: str = constants.DEFAULT_PAIR,
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials: int = constants.DEFAULT_N_TRIALS,
    backtest_features_gcs_path: str = f"{constants.DATA_PATH}/{constants.DEFAULT_PAIR}/{constants.DEFAULT_TIMEFRAME}/{constants.DEFAULT_PAIR}_{constants.DEFAULT_TIMEFRAME}_unseen.parquet",
    vertex_machine_type: str = constants.DEFAULT_VERTEX_LSTM_MACHINE_TYPE,
    vertex_accelerator_type: str = constants.DEFAULT_VERTEX_LSTM_ACCELERATOR_TYPE,
    vertex_accelerator_count: int = constants.DEFAULT_VERTEX_LSTM_ACCELERATOR_COUNT
):
    # Ahora las llamadas a los componentes son más limpias
    
    ingest_task = component_op_factory["data_ingestion"](
        pair=pair,
        timeframe=timeframe,
        project_id=constants.PROJECT_ID,
        polygon_secret_name=constants.POLYGON_API_KEY_SECRET_NAME,
        start_date="2010-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        min_rows=100000,
    )

    prepare_opt_data_task = component_op_factory["data_preparation"](
        pair=pair,
        timeframe=timeframe,
    ).after(ingest_task)

    tuning_task = component_op_factory["hyperparam_tuning"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"],
        pair=pair,
        timeframe=timeframe,
        n_trials=n_trials,
    )
    
    # El lanzador recibe la URI para el job que va a crear
    train_lstm_task = component_op_factory["train_lstm_launcher"](
        vertex_training_image_uri=args.common_image_uri,
        project_id=constants.PROJECT_ID,
        region=constants.REGION,
        pair=pair,
        timeframe=timeframe,
        params_path=tuning_task.outputs["best_params_path"],
        output_gcs_base_dir=constants.LSTM_MODELS_PATH,
        vertex_machine_type=vertex_machine_type,
        vertex_accelerator_type=vertex_accelerator_type,
        vertex_accelerator_count=vertex_accelerator_count,
        vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
    )

    # ... (El resto de las tareas de la pipeline siguen igual)
    prepare_rl_data_task = component_op_factory["prepare_rl_data"](
        lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        pair=pair,
        timeframe=timeframe,
        output_gcs_base_dir=constants.RL_DATA_INPUTS_PATH,
    )

    train_rl_task = component_op_factory["train_rl"](
        params_path=f"{train_lstm_task.outputs['trained_lstm_dir_path']}/params.json",
        rl_data_path=prepare_rl_data_task.outputs["rl_data_path"],
        pair=pair,
        timeframe=timeframe,
        output_gcs_base_dir=constants.RL_MODELS_PATH,
        tensorboard_logs_base_dir=constants.TENSORBOARD_LOGS_PATH,
    )

    backtest_task = component_op_factory["backtest"](
        lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
        features_path=backtest_features_gcs_path,
        pair=pair,
        timeframe=timeframe,
    )

    promotion_task = component_op_factory["model_promotion"](
        new_metrics_dir=backtest_task.outputs["output_gcs_dir"],
        new_lstm_artifacts_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        new_rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
        pair=pair,
        timeframe=timeframe,
        production_base_dir=constants.PRODUCTION_MODELS_PATH,
    )

# --- Compilación y Ejecución de la Pipeline ---
if __name__ == "__main__":
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

        # Ya no necesitamos pasar 'common_image_uri' aquí, porque la pipeline ya no lo tiene como parámetro.
        # Los componentes fueron modificados en memoria antes de la compilación.
        job = aip.PipelineJob(
            display_name=job_display_name,
            template_path=pipeline_filename,
            pipeline_root=constants.PIPELINE_ROOT,
            enable_caching=False,
            parameter_values={
                "pair": constants.DEFAULT_PAIR,
                "timeframe": constants.DEFAULT_TIMEFRAME,
                "n_trials": constants.DEFAULT_N_TRIALS,
            }
        )
        
        print(f"Enviando PipelineJob '{job_display_name}' con la imagen '{args.common_image_uri}'...")
        job.run()
        print(f"✅ PipelineJob enviado. Puedes verlo en la consola de Vertex AI.")