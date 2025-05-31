#!/usr/bin/env python3
"""
model_promotion_decision.py
───────────────────────────
Decide si promocionar un nuevo modelo de trading a producci?n
bas?ndose en m?tricas de backtesting comparadas con el modelo actual en producci?n.

Si el nuevo modelo cumple los criterios, copia sus artefactos
a la ruta de producci?n en GCS y actualiza las m?tricas de producci?n.
Tambi?n publica notificaciones a Pub/Sub.

Entrada (argumentos CLI)
------------------------
--new-metrics-path           Ruta gs:// al metrics.json del nuevo modelo.
--current-production-metrics-path Ruta gs:// al metrics.json del modelo actual en producci?n.
--new-lstm-artifacts-dir     Ruta gs:// a la carpeta versionada del nuevo modelo LSTM.
--new-rl-model-path          Ruta gs:// al archivo .zip del nuevo modelo PPO.
--production-base-dir        Ruta gs:// a la carpeta base de producci?n para los modelos.
--pair                       S?mbolo del par de trading.
--timeframe                  Timeframe de los datos.
"""

import os
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import shutil # Para limpiar directorios temporales
import numpy as np # Para manejo de NaN y operaciones num?ricas

# Reutilizar helpers GCS y Pub/Sub del agente de datos
from google.cloud import storage
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Constantes de Configuraci?n para la Decisi?n de Promoci?n ---
# Estas constantes deben ser ajustadas a tus objetivos de negocio y tolerancia al riesgo.

# Criterios Absolutos M?nimos para cualquier modelo (incluso el primero)
MIN_TRADES_FOR_PROMOTION = 100 # N?mero m?nimo de trades en el backtest
MIN_NET_PIPS_ABSOLUTE = 0.0    # Pips netos positivos
MIN_PROFIT_FACTOR_ABSOLUTE = 1.0 # Profit Factor > 1.0 (ganar m?s de lo que se pierde)
MIN_SHARPE_ABSOLUTE_FIRST_DEPLOY = 0.5 # Sharpe m?nimo si es el primer despliegue

# Umbrales para la funci?n de normalizaci?n de scores (targets)
# Estos son valores de referencia para considerar una m?trica "buena" (1.0 en el score normalizado)
TARGETS = {
    "sharpe": 0.60,    # Sharpe Ratio aceptable
    "pf": 1.75,        # Profit Factor que indica un sistema s?lido
    "winrate": 0.50,   # Win Rate del 50%
    "expectancy": 0.0, # Expectativa positiva por trade
    "trades": 150,     # N?mero de trades para un score alto en "n?mero de trades"
}

# Pesos para el c?lculo del score ponderado final
WEIGHTS = {
    "sharpe": 0.35,
    "dd":     0.30,
    "pf":     0.25,
    "win":    0.05,
    "exp":    0.03,
    "ntrades":0.02,
}

# Umbrales para el veto independiente (reglas de supervivencia)
MAX_DRAWDOWN_REL_TOLERANCE_FACTOR = 1.5 # El nuevo DD absoluto no debe ser > 1.5 * DD absoluto actual
MAX_DRAWDOWN_ABS_THRESHOLD = 25.0 # El nuevo DD absoluto no debe superar 25.0 pips absolutos (o ATRs)
                                  # Asegurarse de que las m?tricas de DD sean en pips o ATRs consistentes.

# Umbral global para PROMOCIONAR el modelo basado en el score ponderado
GLOBAL_PROMOTION_SCORE_THRESHOLD = 0.80

# --- Pub/Sub Topics (Reutilizamos los del Agente de Datos para notificaciones) ---
PUBSUB_PROJECT_ID = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT"))
SUCCESS_TOPIC_ID = "data-ingestion-success" # Usar para notificaci?n de PROMOCION
FAILURE_TOPIC_ID = "data-ingestion-failures" # Usar para notificaci?n de NO PROMOCION

# --- Helpers GCS (Reutilizados de otros scripts) ---
def get_gcs_client():
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            creds = service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            )
            return storage.Client(credentials=creds)
        except Exception as e:
            logger.error(f"Error al cargar credenciales de GOOGLE_APPLICATION_CREDENTIALS: {e}")
            pass
    try:
        return storage.Client()
    except DefaultCredentialsError as e:
        logger.error(f"No se pudieron obtener credenciales predeterminadas de GCP: {e}.")
        raise

def download_gs(uri: str, dest_dir: Path) -> Path:
    """Descarga un archivo de GCS a un directorio especificado."""
    bucket_name, blob_name = uri[5:].split("/", 1)
    local_path = dest_dir / Path(blob_name).name
    get_gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    logger.info(f"📥 Descargado {uri} a {local_path}")
    return local_path

def upload_gs_file(local_path: Path, gcs_uri: str):
    """Sube un archivo local a GCS."""
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    get_gcs_client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
    logger.info(f"☁️ Subido {local_path.name} a {gcs_uri}")

def copy_gcs_object(source_uri: str, destination_uri: str):
    """Copia un objeto en GCS."""
    client = get_gcs_client()
    src_bucket_name, src_blob_name = source_uri[5:].split("/", 1)
    dst_bucket_name, dst_blob_name = destination_uri[5:].split("/", 1)

    source_bucket = client.bucket(src_bucket_name)
    source_blob = source_bucket.blob(src_blob_name)
    
    destination_bucket = client.bucket(dst_bucket_name)
    
    source_bucket.copy_blob(source_blob, destination_bucket, dst_blob_name)
    logger.info(f"? Copiado {source_uri} a {destination_uri}")

def load_metrics(gcs_path: str, is_production_metrics: bool = False) -> dict:
    """Carga m?tricas desde un archivo JSON en GCS."""
    temp_dir = Path(tempfile.mkdtemp()) # Directorio temporal para la descarga
    try:
        local_path = download_gs(gcs_path, temp_dir)
        with open(local_path, 'r') as f:
            metrics = json.load(f)
        logger.info(f"M?tricas cargadas de {gcs_path}")
        # Limpiar el tempdir
        shutil.rmtree(temp_dir)
        return metrics['filtered'] # Interesan las m?tricas de la estrategia FILTRADA
    except Exception as e:
        # Si el archivo de producci?n no existe, inicializar con valores "malos"
        if is_production_metrics:
            logger.warning(f"No se pudo cargar metrics_production.json desde {gcs_path}: {e}. Asumiendo valores iniciales para primer despliegue.")
            # Valores muy bajos/negativos para que el primer modelo siempre gane la comparaci?n
            return {
                "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": -1e9, "net_pips": -1e9, "sharpe": -1e9,
                "sortino": -1e9, "max_drawdown": -1e9, "avg_mfe": 0.0, "avg_mae": 0.0
            }
        else:
            logger.critical(f"Error cr?tico: No se pudieron cargar las m?tricas del NUEVO modelo desde {gcs_path}: {e}")
            # Limpiar el tempdir antes de relanzar
            shutil.rmtree(temp_dir)
            raise # No se puede continuar sin las m?tricas del nuevo modelo

# --- Pub/Sub Helper para Notificaciones ---
PUBSUB_PROJECT_ID = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT"))

def get_pubsub_publisher_client():
    return pubsub_v1.PublisherClient()

def publish_notification(topic_id: str, status: str, message: str, details: dict = None):
    publisher = get_pubsub_publisher_client()
    topic_path = publisher.topic_path(PUBSUB_PROJECT_ID, topic_id)
    
    notification_data = {
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details if details is not None else {}
    }
    message_json = json.dumps(notification_data)
    message_bytes = message_json.encode("utf-8")

    try:
        future = publisher.publish(topic_path, message_bytes)
        message_id = future.result()
        logger.info(f"Notificaci?n publicada en {topic_id} con ID: {message_id}")
    except Exception as e:
        logger.error(f"Error al publicar notificaci?n en {topic_id}: {e}")

# --- Funci?n de clip para normalizaci?n de scores ---
def clip(x):
    return max(0, min(x, 1))

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Decide si promocionar un nuevo modelo de trading a producci?n.")
    parser.add_argument("--new-metrics-path", required=True, help="Ruta gs:// al metrics.json del nuevo modelo (salida de backtest.py).")
    parser.add_argument("--current-production-metrics-path", required=True, help="Ruta gs:// al metrics.json del modelo actual en producci?n.")
    parser.add_argument("--new-lstm-artifacts-dir", required=True, help="Ruta gs:// a la carpeta versionada del nuevo modelo LSTM.")
    parser.add_argument("--new-rl-model-path", required=True, help="Ruta gs:// al archivo .zip del nuevo modelo PPO.")
    parser.add_argument("--production-base-dir", required=True, help="Ruta gs:// a la carpeta base de producci?n para los modelos.")
    parser.add_argument("--pair", required=True, help="S?mbolo del par de trading.")
    parser.add_argument("--timeframe", required=True, help="Timeframe de los datos.")
    args = parser.parse_args()

    logger.info(f"[{datetime.utcnow().isoformat()} UTC] Iniciando decisi?n de promoci?n para {args.pair} | {args.timeframe}")

    # 1. Cargar m?tricas
    try:
        new_metrics_full = load_metrics(args.new_metrics_path, is_production_metrics=False)
        current_metrics_full = load_metrics(args.current_production_metrics_path, is_production_metrics=True)
    except Exception: # load_metrics ya loguea el error y relanza, aqu? solo capturamos para no salir abruptamente
        publish_notification(FAILURE_TOPIC_ID, "ERROR", 
                             f"Fallo cr?tico en la decisi?n de promoci?n para {args.pair} {args.timeframe}: Error al cargar m?tricas.",
                             {"new_metrics_path": args.new_metrics_path, "current_production_metrics_path": args.current_production_metrics_path})
        sys.exit(1)

    new_metrics = new_metrics_full # Alias para claridad
    current_metrics = current_metrics_full # Alias para claridad

    logger.info(f"M?tricas del nuevo modelo: {json.dumps(new_metrics, indent=2)}")
    logger.info(f"M?tricas del modelo actual en producci?n: {json.dumps(current_metrics, indent=2)}")

    promotion_reasons = [] 
    
    # --- 2. Aplicar Veto Independiente (Regla de Supervivencia) ---
    is_vetoed = False
    
    # Calcular Max Drawdown Absoluto para el nuevo y el actual
    new_dd_abs = abs(new_metrics.get('max_drawdown', -1e9)) # Usar .get para seguridad, -1e9 para DD muy malo
    current_dd_abs = abs(current_metrics.get('max_drawdown', -1e9))

    # Primer Veto: Si el nuevo DD absoluto es demasiado grande, sin importar el actual.
    if new_dd_abs > MAX_DRAWDOWN_ABS_THRESHOLD and MAX_DRAWDOWN_ABS_THRESHOLD > 0: # Solo si el umbral no es cero
        is_vetoed = True
        promotion_reasons.append(f"VETO: El Max Drawdown ({new_dd_abs:.2f}) supera el umbral absoluto ({MAX_DRAWDOWN_ABS_THRESHOLD:.2f}).")
    
    # Segundo Veto: Si el nuevo DD absoluto empeora demasiado respecto al actual (solo si hay un modelo actual).
    if current_dd_abs > 0 and not is_vetoed: # Evitar div by zero y solo si no ha sido vetado ya
        if new_dd_abs >= current_dd_abs * MAX_DRAWDOWN_REL_TOLERANCE_FACTOR:
            is_vetoed = True
            promotion_reasons.append(f"VETO: El Max Drawdown ({new_dd_abs:.2f}) empeora m?s de {MAX_DRAWDOWN_REL_TOLERANCE_FACTOR}x respecto al actual ({current_dd_abs:.2f}).")

    if is_vetoed:
        logger.warning(f"⚠️ Modelo {args.pair} {args.timeframe} VETADO. No se desplegar?.")
        for reason in promotion_reasons:
            logger.warning(f" - Raz?n del Veto: {reason}")
        
        publish_notification(FAILURE_TOPIC_ID, "VETADO",
                             f"Modelo {args.pair} {args.timeframe} VETADO. Riesgo inaceptable.",
                             {"new_metrics": new_metrics, "current_metrics": current_metrics, "reasons": promotion_reasons})
        sys.exit(0) # Salir con ?xito, pero el modelo no se promovi?

    # --- 3. Calcular Scores Normalizados y Ponderados ---
    # Manejar el caso de div by zero si las m?tricas actuales son muy malas (ej. Profit Factor = 0)
    
    # Sharpe Score
    sharpe_score = clip(new_metrics.get('sharpe', 0.0) / TARGETS["sharpe"])
    
    # Profit Factor Score
    new_pf = new_metrics.get('profit_factor', 0.0)
    pf_score = clip(new_pf / TARGETS["pf"]) if TARGETS["pf"] > 0 else 0.0
    
    # Win Rate Score
    winrate_score = clip(new_metrics.get('win_rate', 0.0) / TARGETS["winrate"])
    
    # Expectancy Score
    new_exp = new_metrics.get('expectancy', 0.0)
    exp_score = clip((new_exp - TARGETS["expectancy"]) / (abs(new_exp) + 1e-9) + 1) # Asegurar no div by zero
    
    # Number of Trades Score
    new_trades = new_metrics.get('trades', 0)
    trades_score = min(1, (new_trades - TARGETS["trades"]) / TARGETS["trades"]) if TARGETS["trades"] > 0 else 1 # Si trades target es 0, siempre 1.
    
    # Drawdown Score (inverso: menor DD es mejor, 1.0 es perfecto, 0.0 es malo)
    # Se compara con el current_dd_abs para ver si el nuevo DD es "suficientemente bueno" en relaci?n al actual.
    # Un 0.20 de tolerancia significa que si el nuevo DD es un 20% peor, el score es 0.
    # Si new_dd_abs es mucho mejor, el score se clipea a 1.
    dd_score = clip(1 - (new_dd_abs - current_dd_abs) / (current_dd_abs * 0.20 + 1e-9)) if current_dd_abs > 0 else 1.0 # Si no hay current_dd_abs, es 1.0

    # Calcular score ponderado
    total_score = (
        sharpe_score * WEIGHTS["sharpe"] +
        dd_score     * WEIGHTS["dd"]     +
        pf_score     * WEIGHTS["pf"]     +
        winrate_score * WEIGHTS["win"]    +
        exp_score    * WEIGHTS["exp"]    +
        trades_score * WEIGHTS["ntrades"]
    )
    
    logger.info(f"Scores Normalizados: Sharpe={sharpe_score:.2f}, PF={pf_score:.2f}, WinRate={winrate_score:.2f}, Exp={exp_score:.2f}, Trades={trades_score:.2f}, DD={dd_score:.2f}")
    logger.info(f"Score Ponderado Final: {total_score:.2f} (Umbral de promoci?n: {GLOBAL_PROMOTION_SCORE_THRESHOLD:.2f})")

    # --- 4. Decisi?n Final ---
    is_promotable = (total_score >= GLOBAL_PROMOTION_SCORE_THRESHOLD)

    if is_promotable:
        logger.info("✨ El nuevo modelo cumple los criterios de promoci?n. Iniciando despliegue a producci?n...")
        
        # Copiar artefactos a producci?n
        # LSTM Model
        copy_gcs_object(
            f"{args.new_lstm_artifacts_dir}/model.h5",
            f"{args.production_base_dir}/model.h5"
        )
        copy_gcs_object(
            f"{args.new_lstm_artifacts_dir}/scaler.pkl",
            f"{args.production_base_dir}/scaler.pkl"
        )
        copy_gcs_object(
            f"{args.new_lstm_artifacts_dir}/params.json",
            f"{args.production_base_dir}/params.json"
        )
        # RL Model
        copy_gcs_object(
            args.new_rl_model_path, # Esto es un archivo .zip directo
            f"{args.production_base_dir}/ppo_filter_model.zip"
        )

        # Actualizar metrics_production.json
        # Guardar el json de las nuevas m?tricas en un tempfile local
        with tempfile.TemporaryDirectory() as tmpdir:
            local_metrics_file = Path(tmpdir) / "metrics_production.json"
            json.dump({'filtered': new_metrics}, open(local_metrics_file, "w"), indent=2) # Guardar en el formato esperado
            upload_gs_file(local_metrics_file, f"{args.production_base_dir}/metrics_production.json")
            
        logger.info("✅ Modelo desplegado exitosamente a producci?n y m?tricas actualizadas.")
        publish_notification(SUCCESS_TOPIC_ID, "SUCCESS", 
                             f"Modelo {args.pair} {args.timeframe} PROMOVIDO a producci?n.",
                             {"new_metrics": new_metrics, "current_metrics": current_metrics, "score": total_score})

    else:
        logger.warning("⚠️ El nuevo modelo NO cumple los criterios de promoci?n. No se desplegar?.")
        for reason in promotion_reasons: # Si no se vet? al principio, estas son las razones de no cumplir el score
            logger.warning(f" - Raz?n: {reason}")
        
        publish_notification(FAILURE_TOPIC_ID, "NO_PROMOTED",
                             f"Modelo {args.pair} {args.timeframe} NO PROMOVIDO. Score: {total_score:.2f}.",
                             {"new_metrics": new_metrics, "current_metrics": current_metrics, "score": total_score, "reasons": promotion_reasons})

    logger.info(f"[{datetime.utcnow().isoformat()} UTC] Fin de la decisi?n de promoci?n.")


if __name__ == "__main__":
    main()