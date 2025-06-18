# src/pipeline/main.py
"""
Pipeline v5 – Final. Reemplaza el filtro RL por un clasificador supervisado (LightGBM).
Implementa una gestión de rutas centralizada y robusta.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import google.cloud.aiplatform as aip
from kfp import compiler, dsl
from kfp.dsl import ConcatPlaceholder  # type: ignore
from kfp.components import load_component_from_text

# AJUSTE: Se asegura que 'constants' sea la única fuente de configuración.
from src.shared import constants

# ───────────────────────────── CLI ──────────────────────────────
parser = argparse.ArgumentParser(
    "Compila y/o envía la pipeline v5 final con filtro supervisado"
)
parser.add_argument(
    "--common-image-uri",
    required=True,
    help="URI Docker – MISMA imagen para **todos** los componentes",
)
args, _ = parser.parse_known_args()

# ─────────────────────── Carga de Componentes ─────────────────────
COMPONENTS_DIR = Path(__file__).parent.parent / "components"


def load_utf8_component(rel_path: str):
    """Carga un componente YAML preservando UTF-8 (Windows-safe)."""
    yaml_text = (COMPONENTS_DIR / rel_path).read_text(encoding="utf-8")
    return load_component_from_text(yaml_text)


# Cargar todos los componentes necesarios para la v5
component_op_factory = {
    "data_ingestion": load_utf8_component("data_ingestion/component.yaml"),
    "data_preparation": load_utf8_component("data_preparation/component.yaml"),
    "optimize_model_architecture": load_utf8_component(
        "optimize_model_architecture/component.yaml"
    ),
    "optimize_trading_logic": load_utf8_component(
        "optimize_trading_logic/component.yaml"
    ),
    "train_lstm_launcher": load_utf8_component("train_lstm_launcher/component.yaml"),
    "train_filter_model": load_utf8_component("train_filter_model/component.yaml"),
    "backtest": load_utf8_component("backtest/component.yaml"),
    "model_promotion": load_utf8_component("model_promotion/component.yaml"),
}

# Asignar la misma imagen Docker a todos los contenedores para simplificar
for comp in component_op_factory.values():
    impl = comp.component_spec.implementation
    if hasattr(impl, "container") and impl.container:
        impl.container.image = args.common_image_uri

# ───────────────────────── Definición de la Pipeline ─────────────────────────
@dsl.pipeline(
    name="algo-trading-mlops-pipeline-v5-robust-paths",
    description="Versión final con gestión de rutas centralizada, versionada y robusta.",
    pipeline_root=constants.PIPELINE_ROOT,
)
def trading_pipeline_v5(
    timeframe: str = constants.DEFAULT_TIMEFRAME,
    n_trials_arch: int = 20,
    n_trials_logic: int = 30,
    backtest_years_to_keep: int = 5,
    holdout_months: int = 3,
):
    """Pipeline de entrenamiento y despliegue para IA de trading Forex."""

    # 1 ▸ Ingestión de datos
    ingest_task = component_op_factory["data_ingestion"](
        project_id=constants.PROJECT_ID,
        polygon_secret_name=constants.POLYGON_API_KEY_SECRET_NAME,
        end_date=datetime.utcnow().strftime("%Y-%m-%d"),
        timeframe=timeframe,
    )

    # 2 ▸ Preparación de datos (con hold-out)
    prepare_opt_data_task = component_op_factory["data_preparation"](
        pair="ALL",
        timeframe=timeframe,
        years_to_keep=backtest_years_to_keep,
        holdout_months=holdout_months,
    ).after(ingest_task)

    # 3 ▸ Optimizar arquitectura LSTM
    optimize_arch_task = component_op_factory["optimize_model_architecture"](
        features_path=prepare_opt_data_task.outputs["prepared_data_path"],
        n_trials=n_trials_arch,
    )

    # ───────── Bucle paralelo por cada par de divisas ─────────
    pairs_to_process = list(constants.SPREADS_PIP.keys())

    with dsl.ParallelFor(
        items=pairs_to_process, name="parallel-training-for-each-pair"
    ) as pair:

        # 4 ▸ Optimizar lógica de trading (umbrales, hiper-parámetros, etc.)
        # Esta tarea ahora se ejecuta dentro del bucle para acceder a `pair`.
        optimize_logic_task = component_op_factory["optimize_trading_logic"](
            features_path=prepare_opt_data_task.outputs["prepared_data_path"],

            # --- CORRECCIÓN: KFP v2 utiliza `ConcatPlaceholder` para
            # interpolar dinámicamente dentro de un `ParallelFor`.
            architecture_params_file=ConcatPlaceholder([
                optimize_arch_task.outputs["best_architecture_dir"],
                "/",
                pair,
                "/best_architecture.json",
            ]),

            n_trials=n_trials_logic,
        )

        # 5 ▸ Entrenamiento LSTM para el par actual
        train_lstm_task = component_op_factory["train_lstm_launcher"](
            project_id=constants.PROJECT_ID,
            region=constants.REGION,
            pair=pair,
            timeframe=timeframe,
            # --- CORRECCIÓN: Ruta dinámica construida con `ConcatPlaceholder` ---
            params_file=ConcatPlaceholder([
                optimize_logic_task.outputs["best_params_dir"],
                "/",
                pair,
                "/best_params.json",
            ]),
            features_gcs_path=prepare_opt_data_task.outputs["prepared_data_path"],
            output_gcs_base_dir=constants.LSTM_MODELS_PATH,
            vertex_training_image_uri=args.common_image_uri,
            vertex_machine_type=constants.DEFAULT_VERTEX_GPU_MACHINE_TYPE,
            vertex_accelerator_type=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE,
            vertex_accelerator_count=constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT,
            vertex_service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
        )

        # 6 ▸ Entrenamiento del modelo filtro supervisado (LightGBM)
        train_filter_task = component_op_factory["train_filter_model"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            features_path=prepare_opt_data_task.outputs["prepared_data_path"],
            pair=pair,
            timeframe=timeframe,
            output_gcs_base_dir=constants.FILTER_MODELS_PATH,
        )

        # 7 ▸ Backtest final sobre el hold-out
        backtest_task = component_op_factory["backtest"](
            lstm_model_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            filter_model_path=train_filter_task.outputs["trained_filter_model_path"],
            features_path=prepare_opt_data_task.outputs["holdout_data_path"],
            pair=pair,
            timeframe=timeframe,
        )
        
        # 8 ▸ Promoción a producción si el backtest pasa los umbrales
        component_op_factory["model_promotion"](
            new_metrics_dir=backtest_task.outputs["output_gcs_dir"],
            new_lstm_artifacts_dir=train_lstm_task.outputs["trained_lstm_dir_path"],
            new_filter_model_path=train_filter_task.outputs["trained_filter_model_path"],
            pair=pair,
            timeframe=timeframe,
            production_base_dir=constants.PRODUCTION_MODELS_PATH,
        ).after(backtest_task)


    # Recursos recomendados para steps de Optuna y Backtest (fuera del bucle)
    for task in (optimize_arch_task, optimize_logic_task, backtest_task):
        task.set_cpu_limit("8") \
            .set_memory_limit("30G") \
            .set_accelerator_limit(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT) \
            .set_accelerator_type(constants.DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE)


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PIPELINE_JSON = "algo_trading_mlops_pipeline_v5_corrected.json"

    # 1 ▸ Compilar
    compiler.Compiler().compile(trading_pipeline_v5, PIPELINE_JSON)
    print(f"✅ Pipeline v5 (rutas corregidas) compilada a {PIPELINE_JSON}")

    # 2 ▸ Enviar a Vertex AI (si la variable de entorno lo permite)
    if os.getenv("SUBMIT_PIPELINE_TO_VERTEX", "true").lower() == "true":
        aip.init(project=constants.PROJECT_ID, location=constants.REGION)
        display_name = f"algo-trading-v5-robust-{datetime.utcnow():%Y%m%d-%H%M%S}"
        
        job = aip.PipelineJob(
            display_name=display_name,
            template_path=PIPELINE_JSON,
            pipeline_root=constants.PIPELINE_ROOT,
            enable_caching=True
        )
        job.run()
        print(f"🚀 Pipeline lanzada con Display Name: {display_name}")
    else:
        print("⏭️ La pipeline no se envió a Vertex AI (SUBMIT_PIPELINE_TO_VERTEX está en 'false').")
