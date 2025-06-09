# src/components/train_rl/task.py
"""
Tarea del componente de entrenamiento del agente de Reinforcement Learning (PPO).

Responsabilidades:
1.  Cargar los hiperparámetros del archivo `params.json`.
2.  Cargar el dataset para RL (archivo .npz).
3.  Definir un entorno de `gymnasium` (`SignalFilterEnv`).
4.  Configurar y entrenar un agente PPO de `stable-baselines3`.
5.  Guardar el modelo RL entrenado (.zip) en GCS.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.shared import constants, gcs_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("🚀 GPU(s) detectadas y configuradas.")
    else:
        logger.info("ℹ️ No se detectaron GPUs. Se ejecutará en CPU.")
except Exception as e:
    logger.warning(f"⚠️ No se pudo configurar la GPU: {e}")

class SignalFilterEnv(Env):
    metadata = {"render.modes": []}

    def __init__(self, obs: np.ndarray, raw_pnl_signal: np.ndarray, penalty: float):
        super().__init__()
        self._obs = obs
        self._pnl = raw_pnl_signal
        self._penalty = penalty
        self._max_steps = len(obs) - 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs.shape[1],), dtype=np.float32
        )
        self._current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        return self._obs[0], {}

    def step(self, action: int):
        is_accepted = (action == 1)
        reward = self._pnl[self._current_step] if is_accepted else 0.0
        if is_accepted and reward == 0.0:
            reward = self._penalty
        self._current_step += 1
        done = self._current_step > self._max_steps
        next_obs = self._obs[self._current_step] if not done else np.zeros_like(self._obs[0])
        return next_obs, reward, done, False, {}

def run_rl_training(
    params_path: str,
    rl_data_path: str,
    pair: str,
    timeframe: str,
    output_gcs_base_dir: str,
    tensorboard_logs_base_dir: str,
) -> str:
    try:
        logger.info(f"Cargando hiperparámetros desde: {params_path}")
        local_params_path = gcs_utils.ensure_gcs_path_and_get_local(params_path)
        hp = json.loads(local_params_path.read_text())

        logger.info(f"Cargando datos de RL desde: {rl_data_path}")
        local_rl_data_path = gcs_utils.ensure_gcs_path_and_get_local(rl_data_path)
        npz_data = np.load(local_rl_data_path)
        obs_data = npz_data["obs"].astype(np.float32)
        raw_pnl_data = npz_data["raw"].astype(np.float32)
        
        if not (len(obs_data) == len(raw_pnl_data) > 0):
            raise ValueError("Los arrays 'obs' y 'raw' están vacíos o tienen longitudes distintas.")
        logger.info(f"✔ Datos de RL cargados. Shape de observaciones: {obs_data.shape}")

        logger.info(f"🏋️  Entrenando agente PPO para {pair}/{timeframe}...")
        penalty = hp.get("ppo_hyperparameters", {}).get("penalty_useless_action", -0.001)
        env = DummyVecEnv([lambda: SignalFilterEnv(obs_data, raw_pnl_data, penalty)])
        
        ppo_cfg = hp.get("ppo_hyperparameters", {})
        total_timesteps = int(hp.get("total_timesteps", 500_000))
        
        tb_log_path = f"{tensorboard_logs_base_dir.rstrip('/')}/{pair}/{timeframe}/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"Logs de TensorBoard se guardarán en: {tb_log_path}")

        model = PPO(
            policy=ppo_cfg.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=ppo_cfg.get("learning_rate", 2e-4),
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 512),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.0),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            seed=SEED,
            verbose=1,
            tensorboard_log=tb_log_path
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        logger.info("✅ Entrenamiento PPO finalizado.")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        final_gcs_path = (
            f"{output_gcs_base_dir.rstrip('/')}/{pair}/{timeframe}/"
            f"{timestamp}/ppo_filter_model.zip"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_path = Path(tmpdir) / "ppo_filter_model.zip"
            model.save(local_model_path)
            gcs_utils.upload_gcs_file(local_model_path, final_gcs_path)

        logger.info(f"🎉 Tarea completada. Modelo RL disponible en: {final_gcs_path}")
        gc.collect()
        return final_gcs_path

    except Exception as e:
        logger.critical(f"❌ Fallo crítico en el entrenamiento RL: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Entrenamiento del Agente RL (PPO).")
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--rl-data-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-gcs-base-dir", default=constants.RL_MODELS_PATH)
    parser.add_argument("--tensorboard-logs-base-dir", default=constants.TENSORBOARD_LOGS_PATH)
    parser.add_argument("--trained-rl-model-path-output", type=Path, required=True)
    
    args = parser.parse_args()

    final_model_path = run_rl_training(
        params_path=args.params_path,
        rl_data_path=args.rl_data_path,
        pair=args.pair,
        timeframe=args.timeframe,
        output_gcs_base_dir=args.output_gcs_base_dir,
        tensorboard_logs_base_dir=args.tensorboard_logs_base_dir,
    )
    
    args.trained_rl_model_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.trained_rl_model_path_output.write_text(final_model_path)