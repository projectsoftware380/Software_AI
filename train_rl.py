#!/usr/bin/env python3
"""
Entrena un filtro PPO que decide aceptar/rechazar se?ales basadas en las
predicciones de un modelo LSTM. 100 % autocontenible y listo para Vertex AI.
"""

import os, sys, random, warnings, json, tempfile, gc
from pathlib import Path
import argparse
from datetime import datetime

# ── 1. imports principales ─────────────────────────────────
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan # type: ignore
import pandas as pd, joblib
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import gymnasium as gym
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 👉 importaci?n de indicadores centralizados
# Importaci?n corregida para m?dulos de core
from indicators import build_indicators

# reproducibilidad
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        # Correcci?n: Usar tf.keras.mixed_precision
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("🚀 GPU(s) configuradas y pol?tica de precisi?n mixta establecida.")
    except RuntimeError as e:
        print(f"Error al configurar GPUs o precisi?n mixta: {e}")
        # Considerar si se debe continuar solo con CPU o salir.
else:
    print("ℹ️ No se detectaron GPUs, se usar? CPU.")

# ── 2. utilidades Cloud Storage ─────────────────────────────
def gcs_client():
    """
    Si GOOGLE_APPLICATION_CREDENTIALS est? definida, utiliza esas credenciales
    para acceder a Google Cloud Storage.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        return storage.Client(credentials=creds)
    return storage.Client()

def download_gs(uri: str) -> Path:
    """
    Descarga el archivo de GCS a una ruta local temporal.
    """
    bucket_name, blob_name = uri[5:].split("/", 1)
    temp_dir = Path(tempfile.mkdtemp())
    local_path = temp_dir / Path(blob_name).name
    
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Crear directorios padres si no existen
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    blob.download_to_filename(str(local_path))
    print(f"📥 Descargado {uri} a {local_path}")
    return local_path

def upload_gs(local: Path, uri: str):
    """
    Subir archivo local a GCS.
    """
    bucket_name, blob_name = uri[5:].split("/", 1)
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local))
    print(f"☁️ Subido {local} a {uri}")

def local_or_gs(p: str) -> Path:
    """
    Devuelve la ruta local o descarga desde GCS si es necesario.
    """
    if p.startswith("gs://"):
        return download_gs(p)
    else:
        local_path = Path(p)
        if not local_path.exists():
            raise FileNotFoundError(f"Archivo local no encontrado: {p}")
        return local_path

# ── 3. CLI (ajustado para entradas y salidas) ──────────────────────────
cli = argparse.ArgumentParser(description="Entrena un filtro PPO para se?ales de trading.")
cli.add_argument("--model", required=True, help="Ruta gs:// al modelo LSTM entrenado (.h5).")
cli.add_argument("--scaler", required=True, help="Ruta gs:// al scaler entrenado (.pkl).")
cli.add_argument("--params", required=True, help="Ruta gs:// al archivo JSON de hiperpar?metros del LSTM.")
cli.add_argument("--rl-data", required=True, help="Ruta gs:// al archivo .npz con datos (obs, raw, closes) para PPO (salida de prepare_rl_data.py).")
cli.add_argument("--output-bucket", type=str, default="trading-ai-models-460823", help="Bucket de GCS para subir artefactos.")
cli.add_argument("--tensorboard-log-dir", type=str, default=None, help="Ruta gs:// para guardar logs de TensorBoard.")

args = cli.parse_args()

# ── 4. hiperpar?metros JSON ───────────────────────────────
print("⚙️ Cargando hiperpar?metros...")
hp_path = local_or_gs(args.params)
hp = json.loads(hp_path.read_text())
PAIR, TF = hp["pair"], hp["timeframe"]
tick = 0.01 if PAIR.endswith("JPY") else 0.0001 # Asumiendo que 'tick' es globalmente ?til
ATR_LEN = hp.get("atr_len", 14) # Usar .get para default si no est? en hp

# ── 5. modelo LSTM y scaler ───────────────────────────────
print("🧠 Cargando modelo LSTM y scaler...")
lstm_model_path = local_or_gs(args.model)
lstm_model = tf.keras.models.load_model(lstm_model_path, compile=False)
scaler_path = local_or_gs(args.scaler)
scaler = joblib.load(scaler_path)

try:
    # Asegurarse de que la capa de embedding exista y sea accesible
    if len(lstm_model.layers) < 2:
        raise ValueError("El modelo LSTM tiene muy pocas capas para extraer el embedding.")
    emb_model = tf.keras.Model(inputs=lstm_model.input, outputs=lstm_model.layers[-2].output)
except Exception as e:
    print(f"Error al crear emb_model: {e}")
    print("Aseg?rate de que la arquitectura del modelo LSTM es compatible y la capa [-2] es la capa de embedding deseada.")
    sys.exit(1)


# ────────── 6. Cargar datos de entrada del PPO (.npz) ──────────
# Los datos OBS, PNL_TRUE (se?ales o PnL precalculado), y closes vienen de prepare_rl_data.py v?a .npz
print(f"💾 Cargando datos preprocesados para PPO desde: {args.rl_data}")
rl_data_path = local_or_gs(args.rl_data)
try:
    rl_data = np.load(rl_data_path)
    OBS = rl_data['obs'].astype(np.float32)   # Este OBS es el que se usar?
    PNL_TRUE = rl_data['raw'].astype(np.float32) # PNL_TRUE aqu? es la se?al cruda o PnL precalculado
    closes_from_npz = rl_data['closes'].astype(np.float32)
    print(f"Cargado OBS shape: {OBS.shape}, PNL_TRUE shape: {PNL_TRUE.shape}, closes_from_npz shape: {closes_from_npz.shape}")
    if OBS.shape[0] == 0 or PNL_TRUE.shape[0] == 0 or closes_from_npz.shape[0] == 0:
        print("🚨 Error: Uno de los arrays cargados (obs, raw, closes) est? vac?o.")
        sys.exit(1)
    if OBS.shape[0] != PNL_TRUE.shape[0] or OBS.shape[0] != closes_from_npz.shape[0]:
        print("🚨 Error: Las longitudes de los arrays cargados (obs, raw, closes) no coinciden.")
        sys.exit(1)

except FileNotFoundError:
    print(f"🚨 Error: Archivo .npz no encontrado en {rl_data_path}")
    sys.exit(1)
except KeyError as e:
    print(f"🚨 Error: Falta la clave {e} en el archivo .npz. Aseg?rate de que 'obs', 'raw', y 'closes' est?n presentes.")
    sys.exit(1)


# ────────── 7. entorno Gymnasium ────────────────
print("🛠️ Definiendo entorno Gymnasium SignalFilterEnv...")
class SignalFilterEnv(gym.Env):
    def __init__(self, obs_data, pnl_or_signal_data, closes_data):
        super().__init__()
        self.obs_data = obs_data
        self.pnl_or_signal_data = pnl_or_signal_data # Puede ser PnL directo o se?al (1,0,-1)
        self.closes_data = closes_data # Podr?a usarse para calcular PnL si pnl_or_signal_data es solo se?al

        self.current_step = 0
        self.max_steps = len(obs_data) -1 # Index goes from 0 to len-1

        self.action_space = gym.spaces.Discrete(2)  # 0: reject, 1: accept
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_data.shape[1],),
            dtype=np.float32
        )
        # print(f"Entorno inicializado. Max steps: {self.max_steps}, Obs shape: {obs_data.shape[1]}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) # Importante para la reproducibilidad si se usa seed
        self.current_step = 0
        # print(f"Entorno reseteado. current_step: {self.current_step}")
        return self.obs_data[self.current_step], {}

    def step(self, action):
        if self.current_step > self.max_steps:
            # Esto no deber?a ocurrir si done se maneja correctamente
            raise IndexError("current_step ha excedido max_steps.")

        # Asumiendo que pnl_or_signal_data ya es el PnL por aceptar la se?al.
        # Si pnl_or_signal_data es una se?al (1, -1, 0), aqu? se necesitar?a l?gica para calcular el PnL real.
        # Por el comentario original: "# Por tu c?digo original, `PNL_TRUE` ya es un vector de PnL por tick si se acepta."
        # se asume que es PnL.
        reward = self.pnl_or_signal_data[self.current_step] if action == 1 else 0.0
        
        # Penalizaci?n ligera por acci?n in?til (aceptar una se?al que no da PnL ni positivo ni negativo)
        # Esto solo tiene sentido si pnl_or_signal_data es PnL.
        if action == 1 and self.pnl_or_signal_data[self.current_step] == 0.0:
            reward = hp.get("penalty_useless_action", -0.001) # Hacer configurable
        
        self.current_step += 1
        done = self.current_step > self.max_steps
        
        if done:
            next_observation = np.zeros_like(self.obs_data[0]) # Observaci?n dummy al final
        else:
            next_observation = self.obs_data[self.current_step]
            
        # Truncated no se usa expl?citamente aqu?, PPO lo maneja o se puede a?adir si es necesario.
        # print(f"Step: {self.current_step-1}, Action: {action}, Reward: {reward}, Done: {done}")
        return next_observation, reward, done, False, {} # False para truncated

# ────────── 8. entrenamiento PPO ─────────────────────────────────────────
print(f"🏋️ Iniciando entrenamiento PPO para {PAIR} {TF}...")
vec_env = DummyVecEnv([lambda: SignalFilterEnv(OBS, PNL_TRUE, closes_from_npz)])

# Hiperpar?metros del PPO desde el archivo JSON o defaults
ppo_hps = hp.get("ppo_hyperparameters", {})
total_timesteps = hp.get("total_timesteps", 500_000)

ppo = PPO(
    ppo_hps.get("policy", "MlpPolicy"),
    vec_env,
    learning_rate=ppo_hps.get("learning_rate", 2e-4),
    n_steps=ppo_hps.get("n_steps", 2048),
    batch_size=ppo_hps.get("batch_size", 512),
    n_epochs=ppo_hps.get("n_epochs", 10),
    gamma=ppo_hps.get("gamma", 0.99),
    gae_lambda=ppo_hps.get("gae_lambda", 0.95),
    clip_range=ppo_hps.get("clip_range", 0.2),
    ent_coef=ppo_hps.get("ent_coef", 0.0),
    vf_coef=ppo_hps.get("vf_coef", 0.5),
    max_grad_norm=ppo_hps.get("max_grad_norm", 0.5),
    seed=42,
    verbose=1,
    tensorboard_log=args.tensorboard_log_dir # Usar el argumento CLI para la ruta de TensorBoard
)

ppo.learn(total_timesteps=total_timesteps, progress_bar=True)
print("✅ Entrenamiento PPO completado.")

# ── 9. guardar artefactos ───────────────────────────────────────────────
# Crear un directorio temporal local para guardar los artefactos antes de subir
with tempfile.TemporaryDirectory() as tmpdir:
    local_out_dir = Path(tmpdir) / "rl_model_output" # Subdirectorio para los artefactos temporales
    local_out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = local_out_dir / "ppo_filter_model.zip" # Nombre de archivo m?s descriptivo
    ppo.save(zip_path)
    print(f"💾 Modelo PPO guardado localmente en {zip_path}")

    # ── 10. Subida obligatoria a Cloud Storage con versionado por timestamp ──
    timestamp_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    # La ruta base para los modelos entrenados en GCS
    GCS_MODELS_BASE_PATH = f"gs://{args.output_bucket}/models/RL" # Usar el bucket de salida del argumento
    gcs_dest_path = f"{GCS_MODELS_BASE_PATH}/{PAIR}/{TF}/{timestamp_str}/ppo_filter_model.zip"
    
    try:
        upload_gs(zip_path, gcs_dest_path)
        print(f"☁️ Modelo PPO subido a {gcs_dest_path}")
    except Exception as e:
        print(f"🚨 Error al subir el modelo PPO a GCS {gcs_dest_path}: {e}")
        sys.exit(1) # Fallo cr?tico si no se puede subir el modelo

print(f"🎉 Fin del script ? {datetime.utcnow().isoformat()} UTC")
gc.collect() # Limpiar memoria