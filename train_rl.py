#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rl.py
────────────────────────────────────────────────────────────────────────────
Entrena un “filtro” PPO que decide aceptar o rechazar las señales generadas
por un modelo LSTM.  Diseñado para ejecutarse de forma autónoma en Vertex AI
(CustomJob) o en cualquier entorno con acceso a GCS.

Correciones v2025-06-04 💫
────────────────────────────────────────────────────────────────────────────
•  Añadidas importaciones de google-cloud (storage / service_account).
•  Se unificó la inicialización de GPU y mixed-precision.
•  Sanitizado de caracteres y comentarios a UTF-8.
•  Validaciones más estrictas sobre shapes y contenido de los arrays .npz.
•  Limpieza de memoria explícita y mensajes de log más claros.
"""

# ────────────────────────── imports estándar ─────────────────────────────
import argparse
import gc
import json
import os
import random
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple

# ────────────────────────── imports de terceros ──────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from google.oauth2 import service_account
from gymnasium import Env, spaces
from joblib import load as joblib_load
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ────────────────────────── configuración global ─────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# ────────────────────────── GPU & mixed precision ────────────────────────
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("🚀  GPU(s) disponibles y policy mixed-precision activada.")
    else:
        print("ℹ️  No se detectaron GPU; se usará CPU.")
except Exception as e:
    print(f"⚠️  No se pudo configurar GPU/mixed-precision: {e}")

# ────────────────────────── helpers de GCS ───────────────────────────────
def _gcs_client():
    """Devuelve un cliente de GCS respetando GOOGLE_APPLICATION_CREDENTIALS."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        creds = service_account.Credentials.from_service_account_file(creds_path)
        return storage.Client(credentials=creds)
    return storage.Client()

def download_gs(uri: str) -> Path:
    """Descarga `gs://…` a un tmp local y devuelve la ruta local."""
    bucket_name, blob_name = uri[5:].split("/", 1)
    local_path = Path(tempfile.mkdtemp()) / Path(blob_name).name
    _gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    print(f"📥  Descargado {uri} → {local_path}")
    return local_path

def upload_gs(local: Path, uri: str):
    """Sube un archivo local a `gs://…`."""
    bucket_name, blob_name = uri[5:].split("/", 1)
    _gcs_client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local))
    print(f"☁️  Subido {local} → {uri}")

def as_local(path: str) -> Path:
    """Si es GCS descarga; si es local valida existencia."""
    if path.startswith("gs://"):
        return download_gs(path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ruta local no encontrada: {path}")
    return p

# ────────────────────────── CLI ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="Entrena un filtro PPO sobre las señales LSTM")
parser.add_argument("--model",      required=True, help="gs://… modelo LSTM (.h5)")
parser.add_argument("--scaler",     required=True, help="gs://… scaler (.pkl)")
parser.add_argument("--params",     required=True, help="gs://… hiperparámetros JSON")
parser.add_argument("--rl-data",    required=True, help="gs://… datos .npz (prepare_rl_data)")
parser.add_argument("--output-bucket", default="trading-ai-models-460823",
                    help="Bucket GCS donde subir el modelo PPO")
parser.add_argument("--tensorboard-log-dir", default=None,
                    help="gs://… carpeta para logs TensorBoard (opcional)")
args = parser.parse_args()

# ────────────────────────── carga de artefactos ─────────────────────────
hp_path      = as_local(args.params)
hp           = json.loads(hp_path.read_text())
PAIR         = hp["pair"]
TF           = hp["timeframe"]
TICK         = 0.01 if PAIR.endswith("JPY") else 0.0001
ATR_LEN      = hp.get("atr_len", 14)

lstm_model   = tf.keras.models.load_model(as_local(args.model), compile=False)
scaler       = joblib_load(as_local(args.scaler))

# capa de embedding
if len(lstm_model.layers) < 2:  # sanity-check
    sys.exit("❌  El modelo LSTM no tiene suficientes capas para extraer embedding.")
emb_model = tf.keras.Model(lstm_model.input, lstm_model.layers[-2].output)

# ────────────────────────── carga de datos RL (.npz) ─────────────────────
npz_path = as_local(args.rl_data)
try:
    npz = np.load(npz_path)
    OBS       = npz["obs"].astype(np.float32)
    RAW_PNL   = npz["raw"].astype(np.float32)     # señal PnL por tick
    CLOSES    = npz["closes"].astype(np.float32)
except KeyError as e:
    sys.exit(f"❌  Faltan claves en .npz: {e}")
if not (len(OBS) == len(RAW_PNL) == len(CLOSES) > 0):
    sys.exit("❌  Los arrays obs/raw/closes están vacíos o con longitudes distintas.")

# ────────────────────────── entorno Gymnasium ───────────────────────────
class SignalFilterEnv(Env):
    """Entorno mínimo para filtrar señales LSTM con PPO."""
    metadata = {"render.modes": []}

    def __init__(self, obs: np.ndarray, pnl: np.ndarray):
        super().__init__()
        self._obs = obs
        self._pnl = pnl
        self._max = len(obs) - 1
        self.action_space      = spaces.Discrete(2)  # 0-reject, 1-accept
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs.shape[1],), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._obs[0], {}

    def step(self, action: int):
        reward = self._pnl[self._step] if action == 1 else 0.0
        if action == 1 and reward == 0.0:
            reward = hp.get("penalty_useless_action", -0.001)

        self._step += 1
        done = self._step > self._max
        next_obs = np.zeros_like(self._obs[0]) if done else self._obs[self._step]
        return next_obs, reward, done, False, {}

# ────────────────────────── entrenamiento PPO ───────────────────────────
print(f"🏋️  Entrenando PPO para {PAIR}/{TF} …")
env     = DummyVecEnv([lambda: SignalFilterEnv(OBS, RAW_PNL)])
ppo_cfg = hp.get("ppo_hyperparameters", {})
total_ts = int(hp.get("total_timesteps", 500_000))

model = PPO(
    policy           = ppo_cfg.get("policy", "MlpPolicy"),
    env              = env,
    learning_rate    = ppo_cfg.get("learning_rate", 2e-4),
    n_steps          = ppo_cfg.get("n_steps", 2048),
    batch_size       = ppo_cfg.get("batch_size", 512),
    n_epochs         = ppo_cfg.get("n_epochs", 10),
    gamma            = ppo_cfg.get("gamma", 0.99),
    gae_lambda       = ppo_cfg.get("gae_lambda", 0.95),
    clip_range       = ppo_cfg.get("clip_range", 0.2),
    ent_coef         = ppo_cfg.get("ent_coef", 0.0),
    vf_coef          = ppo_cfg.get("vf_coef", 0.5),
    max_grad_norm    = ppo_cfg.get("max_grad_norm", 0.5),
    seed             = SEED,
    verbose          = 1,
    tensorboard_log  = args.tensorboard_log_dir
)

model.learn(total_timesteps=total_ts, progress_bar=True)
print("✅  Entrenamiento PPO finalizado.")

# ────────────────────────── guardado & subida a GCS ──────────────────────
timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
local_dir = Path(tempfile.mkdtemp())
out_file  = local_dir / "ppo_filter_model.zip"
model.save(out_file)

gcs_path  = (f"gs://{args.output_bucket}/models/RL/"
             f"{PAIR}/{TF}/{timestamp}/ppo_filter_model.zip")
upload_gs(out_file, gcs_path)

print(f"🎉  Script finalizado. Modelo disponible en: {gcs_path}")
gc.collect()
