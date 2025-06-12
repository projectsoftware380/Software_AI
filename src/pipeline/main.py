# src/pipeline/main.py
"""Pipeline v3 – ingestión → HPO → LSTM → RL → backtest → promoción.
Lectura UTF-8 explícita para esquivar UnicodeDecodeError en Windows.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import google.cloud.aiplatform as aip
from kfp import compiler
from kfp.components import load_component_from_text
from kfp.dsl import pipeline

from src.shared import constants

# ───────────────────────────── CLI ──────────────────────────────
parser = argparse.ArgumentParser("Compila y/o envía la pipeline")
parser.add_argument("--common-image-uri", required=True,
                    help="URI Docker (misma imagen) para TODOS los componentes")
args, _ = parser.parse_known_args()

COMPONENTS_DIR = Path(__file__).parent.parent / "components"

def load_utf8_component(rel_path: str):
    yaml_text = (COMPONENTS_DIR / rel_path).read_text(encoding="utf-8")
    return load_component_from_text(yaml_text)

component_op_factory = {
    "data_ingestion":      load_utf8_component("data_ingestion/component.yaml"),
    "data_preparation":    load_utf8_component("data_preparation/component.yaml"),
    "hyperparam_tuning":   load_utf8_component("hyperparam_tuning/component.yaml"),
    "train_lstm_launcher": load_utf8_component("train_lstm_launcher/component.yaml"),
    "prepare_rl_data":     load_utf8_component("prepare_rl_data/component.yaml"),
    "train_rl":            load_utf8_component("train_rl/component.yaml"),
    "backtest":            load_utf8_component("backtest/component.yaml"),
    "model_promotion":     load_utf8_component("model_promotion/component.yaml"),
}

# Aplica la misma imagen a todos los contenedores-componente
for op in component_op_factory.values():
    op.component_spec.implementation.container.image = args.common_image_uri

# ───────────────────────── Definición PIPELINE ─────────────────────────
@pipeline(
    name="algo-trading-mlops-modular-pipeline-v3",
    description="Pipeline v3: ingestión → HPO → LSTM → RL → backtest → promoción",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline(
    pair: str = constants.DEFAULT_PAIR,
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials: int = constants.DEFAULT_N_TRIALS,
    backtest_features_gcs_path: str = (
        f"{constants.DATA_PATH}/{constants.DEFAULT_PAIR}/{constants.DEFAULT_TIMEFRAME}/"
        f"{constants.DEFAULT_PAIR}_{constants.DEFAULT_TIMEFRAME}_unseen.parquet"
    )
):
    # 1 ▸ Ingestión ──────────────────────────────────────────────
    ingest_task = component_op_factory["data_ingestion"](
        pair=pair,
        timeframe=timeframe,
        project_id=constants.PROJECT_ID,
        polygon_secret_name=constants.POLYGON_API_KEY_SECRET_NAME,
        start_date="2010-01-01",
        end_date=datetime.utcnow().strftime("%Y-%m-%d"),
        min_rows=100_000,
    )

    # 2 ▸ Datos para HPO ────────────────────────────────────────
    prepare_opt_data_task = component_op_factory["data_preparation"](
        pair=pair,
        timeframe=timeframe,
    ).after(ingest_task)

    # 3 ▸ Optuna HPO (GPU) ──────────────────────────────────────
    tuning_task = component_op_factory["hyperparam_tuning"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"],
        pair=pair,
        timeframe=timeframe,
        n_trials=n_trials,
    )
    (tuning_task
        .set_cpu_limit("8")          # RAM/CPU explícitas
        .set_memory_limit("30G")
        .set_gpu_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT)
        .set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE))

    # 4 ▸ Entrenamiento LSTM (se lanza como CustomJob GPU) ─────
    train_lstm_task = component_op_factory["train_lstm_launcher"](
        vertex_training_image_uri=args.common_image_uri,
        project_id=constants.PROJECT_ID,
        region=constants.REGION,
        pair=pair,
        timeframe=timeframe,
        params_path=tuning_task.outputs["best_params_path"],
        features_gcs_path=prepare_opt_data_task.outputs["prepared_data_path"],
        output_gcs_base_dir=constants.LSTM_MODELS_PATH,
        vertex_machine_type=constants.DEFAULT_VERTEX_GPU_MACHINE_TYPE,
        vertex_accelerator_type=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE,
        vertex_accelerator_count=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT,
        vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
    )

    # 5 ▸ Datos para RL ─────────────────────────────────────────
    prepare_rl_data_task = component_op_factory["prepare_rl_data"](
        lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        pair=pair,
        timeframe=timeframe,
        output_gcs_base_dir=constants.RL_DATA_INPUTS_PATH,
    )

    # 6 ▸ Entrenamiento PPO (GPU) ──────────────────────────────
    train_rl_task = component_op_factory["train_rl"](
        params_path=f"{train_lstm_task.outputs['trained_lstm_dir_path']}/params.json",
        rl_data_path=prepare_rl_data_task.outputs["rl_data_path"],
        pair=pair,
        timeframe=timeframe,
        output_gcs_base_dir=constants.RL_MODELS_PATH,
        tensorboard_logs_base_dir=constants.TENSORBOARD_LOGS_PATH,
    )
    (train_rl_task
        .set_cpu_limit("8")
        .set_memory_limit("20G")
        .set_gpu_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT)        # 🆕
        .set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE)) # 🆕

    # 7 ▸ Backtest ──────────────────────────────────────────────
    backtest_task = component_op_factory["backtest"](
        lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
        features_path=backtest_features_gcs_path,
        pair=pair,
        timeframe=timeframe,
    )

    # 8 ▸ Promoción a Producción ───────────────────────────────
    component_op_factory["model_promotion"](
        new_metrics_dir=backtest_task.outputs["output_gcs_dir"],
        new_lstm_artifacts_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
        new_rl_model_path=train_rl_task.outputs["trained_rl_model_path"],
        pair=pair,
        timeframe=timeframe,
        production_base_dir=constants.PRODUCTION_MODELS_PATH,
    )

# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PIPELINE_JSON = "algo_trading_mlops_modular_pipeline_v3.json"

    compiler.Compiler().compile(trading_pipeline, PIPELINE_JSON)
    print("✅ Pipeline compilada a", PIPELINE_JSON)

    if os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() == "true":
        aip.init(project=constants.PROJECT_ID, location=constants.REGION)
        display_name = f"algo-trading-v3-{constants.DEFAULT_PAIR}-{datetime.utcnow():%Y%m%d-%H%M%S}"
        job = aip.PipelineJob(
            display_name=display_name,
            template_path=PIPELINE_JSON,
            pipeline_root=constants.PIPELINE_ROOT,
            enable_caching=False,
            parameter_values={
                "pair": constants.DEFAULT_PAIR,
                "timeframe": constants.DEFAULT_TIMEFRAME,
                "n_trials": constants.DEFAULT_N_TRIALS,
            },
        )
        print("🚀 Enviando PipelineJob", display_name)
        job.run()
        print("📊 Sigue el progreso en Vertex AI → Pipelines")