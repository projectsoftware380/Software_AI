# src/deploy/cli.py

import argparse
import logging
from pathlib import Path
from kfp.compiler import Compiler
from datetime import datetime # Importar datetime aquí

import sys
from src.pipeline.pipeline_definition import trading_pipeline_v5, load_component_factory
from src.deploy.vertex_ai_runner import run_vertex_pipeline
from src.shared import constants

# Importaciones para ejecución local
from src.components.dukascopy_ingestion.task import run_ingestion
from src.components.data_preparation.task import run_data_preparation
from src.components.optimize_model_architecture.task import run_optimize_model_architecture
from src.components.optimize_trading_logic.task import run_optimize_trading_logic
from pathlib import Path # Ya está importado, pero lo mantengo para claridad


from src.shared.logging_config import setup_logging

setup_logging() # Llamar a la configuración centralizada
logger = logging.getLogger(__name__)

from src.shared import gcs_utils # Añadir esta importación

def _run_local_and_train_on_gcp(args, end_date):
    logger.info("Modo de ejecución: Fases previas locales, entrenamiento en GCP.")

    # Definir rutas de salida locales y de GCS
    local_temp_dir = Path("./temp_local_pipeline_artifacts")
    local_temp_dir.mkdir(exist_ok=True)

    # Rutas de GCS para los artefactos intermedios
    gcs_base_path = f"{constants.BASE_GCS_PATH}/local_run_artifacts/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    ingested_data_gcs_path = f"{gcs_base_path}/ingested_data"
    prepared_data_gcs_path = f"{gcs_base_path}/prepared_data"
    optimized_arch_gcs_path = f"{gcs_base_path}/optimized_architecture"
    optimized_logic_gcs_path = f"{gcs_base_path}/optimized_logic"

    # --- 1. Ejecutar Ingestión de Datos (Local) ---
    logger.info("Ejecutando Ingestión de Datos (Local)...")
    ingested_data_local_path = local_temp_dir / "ingested_data.parquet"
    ingestion_completion_message_local_path = local_temp_dir / "ingestion_completion.txt"
    
    success_ingestion = run_ingestion(
        pair="EURUSD", # Hardcodeado por ahora, se puede parametrizar
        timeframe="m15",
        project_id=constants.PROJECT_ID,
        gcs_data_path=ingested_data_gcs_path, # La tarea de ingestión ya sube a GCS
        end_date_str=end_date,
        output_gcs_path_output=str(ingested_data_local_path), # KFP espera rutas locales para outputs
        completion_message_path=str(ingestion_completion_message_local_path),
    )
    if not success_ingestion:
        logger.error("Fallo en la ingestión de datos local.")
        sys.exit(1)
    logger.info("✅ Ingestión de Datos completada.")

    # --- 2. Ejecutar Preparación de Datos (Local) ---
    logger.info("Ejecutando Preparación de Datos (Local)...")
    prepared_data_local_path = local_temp_dir / "prepared_data.parquet"
    
    success_preparation = run_data_preparation(
        input_data_path=ingested_data_gcs_path, # Leer desde GCS
        years_to_keep=args.backtest_years_to_keep,
        holdout_months=args.holdout_months,
        output_data_path=str(prepared_data_local_path), # KFP espera rutas locales para outputs
    )
    if not success_preparation:
        logger.error("Fallo en la preparación de datos local.")
        sys.exit(1)
    
    # Subir datos preparados a GCS
    gcs_utils.upload_gcs_file(prepared_data_local_path, prepared_data_gcs_path)
    logger.info(f"Datos preparados subidos a GCS: {prepared_data_gcs_path}")
    logger.info("✅ Preparación de Datos completada.")

    # --- 3. Ejecutar Optimización de Arquitectura (Local) ---
    logger.info("Ejecutando Optimización de Arquitectura (Local)...")
    optimized_arch_local_path = local_temp_dir / "best_architecture.json"
    
    success_arch_opt = run_optimize_model_architecture(
        features_path=prepared_data_gcs_path, # Leer desde GCS
        n_trials=args.n_trials_arch,
        pair="EURUSD", # Hardcodeado por ahora
        output_architecture_path=str(optimized_arch_local_path), # KFP espera rutas locales para outputs
    )
    if not success_arch_opt:
        logger.error("Fallo en la optimización de arquitectura local.")
        sys.exit(1)

    # Subir arquitectura optimizada a GCS
    gcs_utils.upload_gcs_file(optimized_arch_local_path, optimized_arch_gcs_path)
    logger.info(f"Arquitectura optimizada subida a GCS: {optimized_arch_gcs_path}")
    logger.info("✅ Optimización de Arquitectura completada.")

    # --- 4. Ejecutar Optimización de Lógica de Trading (Local) ---
    logger.info("Ejecutando Optimización de Lógica de Trading (Local)...")
    optimized_logic_local_path = local_temp_dir / "best_params.json"
    
    success_logic_opt = run_optimize_trading_logic(
        features_path=prepared_data_gcs_path, # Leer desde GCS
        architecture_params_file=f"{optimized_arch_gcs_path}/best_architecture.json", # Leer desde GCS
        n_trials=args.n_trials_logic,
        pair="EURUSD", # Hardcodeado por ahora
        output_params_path=str(optimized_logic_local_path), # KFP espera rutas locales para outputs
    )
    if not success_logic_opt:
        logger.error("Fallo en la optimización de lógica de trading local.")
        sys.exit(1)

    # Subir lógica optimizada a GCS
    gcs_utils.upload_gcs_file(optimized_logic_local_path, optimized_logic_gcs_path)
    logger.info(f"Lógica optimizada subida a GCS: {optimized_logic_gcs_path}")
    logger.info("✅ Optimización de Lógica de Trading completada.")

    # --- 5. Lanzar Entrenamiento LSTM en GCP ---
    logger.info("Lanzando Entrenamiento LSTM en GCP...")
    
    # Crear un archivo JSON temporal para el pipeline de entrenamiento LSTM
    lstm_pipeline_json_path = local_temp_dir / "train_lstm_pipeline.json"
    
    Compiler().compile(
        trading_pipeline_v5, # Usamos la misma definición de pipeline, pero solo lanzaremos el sub-pipeline de entrenamiento
        str(lstm_pipeline_json_path),
        pipeline_parameters={
            "timeframe": args.timeframe,
            "n_trials_arch": args.n_trials_arch, # No se usan directamente, pero son parte de la firma
            "n_trials_logic": args.n_trials_logic, # No se usan directamente, pero son parte de la firma
            "backtest_years_to_keep": args.backtest_years_to_keep, # No se usan directamente, pero son parte de la firma
            "holdout_months": args.holdout_months, # No se usan directamente, pero son parte de la firma
            "end_date": end_date,
            "common_image_uri": args.common_image_uri # La imagen para los componentes de GCP
        }
    )

    # Lanzar el job de Vertex AI para el entrenamiento LSTM
    run_vertex_pipeline(
        pipeline_json_path=str(lstm_pipeline_json_path),
        project_id=constants.PROJECT_ID,
        region=constants.REGION,
        pipeline_root=constants.PIPELINE_ROOT,
        service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
        display_name_prefix="local-run-lstm-training",
    )
    logger.info("✅ Entrenamiento LSTM lanzado en GCP.")

def main():
    parser = argparse.ArgumentParser(description="Herramienta CLI para compilar y desplegar la pipeline de trading.")
    parser.add_argument("--common-image-uri", required=True, help="URI Docker para todos los componentes.")
    parser.add_argument("--compile-only", action="store_true", help="Solo compila la pipeline a JSON, no la despliega.")
    parser.add_argument("--output-json", type=str, default="algo_trading_mlops_pipeline_v5_final.json", help="Nombre del archivo JSON de salida para la pipeline compilada.")
    
    # Parámetros de la pipeline
    parser.add_argument("--timeframe", type=str, default=constants.DEFAULT_TIMEFRAME, help="Timeframe para los datos.")
    parser.add_argument("--n-trials-arch", type=int, default=20, help="Número de trials para optimización de arquitectura.")
    parser.add_argument("--n-trials-logic", type=int, default=30, help="Número de trials para optimización de lógica de trading.")
    parser.add_argument("--backtest-years-to-keep", type=int, default=5, help="Años a mantener para backtesting.")
    parser.add_argument("--holdout-months", type=int, default=3, help="Meses para holdout.")
    
    parser.add_argument("--run-local-and-train-on-gcp", action="store_true", help="Ejecuta las fases de pre-procesamiento localmente y lanza el entrenamiento en GCP.")

    args = parser.parse_args()

    end_date = datetime.utcnow().strftime("%Y-%m-%d") # Generar end_date aquí

    if args.run_local_and_train_on_gcp:
        _run_local_and_train_on_gcp(args, end_date)
    else:
        logger.info(f"Iniciando compilación del pipeline a '{args.output_json}'...")
        try:
            component_op_factory = load_component_factory()
            logger.info(f"Asignando imagen Docker común a todos los componentes: {args.common_image_uri}")
            for name, comp in component_op_factory.items():
                if hasattr(comp.component_spec.implementation, "container"):
                    comp.component_spec.implementation.container.image = args.common_image_uri
            logger.info("✅ Imagen Docker asignada.")

            Compiler().compile(
                trading_pipeline_v5, 
            args.output_json, 
            pipeline_parameters={
                "timeframe": args.timeframe,
                "n_trials_arch": args.n_trials_arch,
                "n_trials_logic": args.n_trials_logic,
                "backtest_years_to_keep": args.backtest_years_to_keep,
                "holdout_months": args.holdout_months,
                "end_date": end_date,
                "common_image_uri": args.common_image_uri
            }
        )
        logger.info(f"✅ Pipeline compilada exitosamente.")

        if not args.compile_only:
            logger.info("Iniciando despliegue del pipeline a Vertex AI...")
            run_vertex_pipeline(
                pipeline_json_path=args.output_json,
                project_id=constants.PROJECT_ID,
                region=constants.REGION,
                pipeline_root=constants.PIPELINE_ROOT,
                service_account=constants.VERTEX_LSTM_SERVICE_ACCOUNT,
            )
        else:
            logger.info("Despliegue omitido (--compile-only activado).")

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

    except Exception as e:
        logger.exception(f"❌ Fallo fatal al compilar o lanzar el pipeline. Error: {e}")
        sys.exit(1) # Asegurar que el proceso termine con un código de error

if __name__ == "__main__":
    main()
