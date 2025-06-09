# src/components/train_rl/task.py
"""
Entrena un agente PPO (Stable-Baselines3) que filtra las señales del LSTM.
Genera un .zip y lo sube a GCS.  La ruta final se escribe en
`/tmp/trained_rl_model.txt` para que KFP la capture.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.shared import constants, gcs_utils

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


class SignalFilterEnv(Env):
    """Entorno Gym para aceptar/rechazar operaciones según PnL esperado."""
    metadata = {"render.modes": []}

    def __init__(self, obs: np.ndarray, raw_pnl: np.ndarray, penalty: float):
        super().__init__()
        self._obs = obs
        self._pnl = raw_pnl
        self._penalty = penalty
        self._max_steps = len(obs) - 1
        self.action_space = spaces.Discrete(2)  # 0 = skip, 1 = aceptar
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs.shape[1],), dtype=np.float32
        )
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._obs[0], {}

    def step(self, action: int):
        accepted = action == 1
        reward = self._pnl[self._step] if accepted else 0.0
        if accepted and reward == 0.0:
            reward = self._penalty
        self._step += 1
        done = self._step > self._max_steps
        next_obs = self._obs[self._step] if not done else np.zeros_like(self._obs[0])
        return next_obs, reward, done, False, {}


# ──────────────────────────── función principal ────────────────────────────
def run_rl_training(
    *,
    params_path: str,
    rl_data_path: str,
    pair: str,
    timeframe: str,
    output_gcs_base_dir: str,
    tensorboard_logs_base_dir: str,
) -> str:
    logger.info("Cargando hiperparámetros: %s", params_path)
    hp = json.loads(gcs_utils.ensure_gcs_path_and_get_local(params_path).read_text())

    logger.info("Cargando dataset RL: %s", rl_data_path)
    npz = np.load(gcs_utils.ensure_gcs_path_and_get_local(rl_data_path))
    obs, raw = npz["obs"].astype(np.float32), npz["raw"].astype(np.float32)

    if len(obs) != len(raw):
        raise ValueError("Longitudes distintas entre 'obs' y 'raw'")

    penalty = hp.get("ppo_hyperparameters", {}).get("penalty_useless_action", -0.001)
    env = DummyVecEnv([lambda: SignalFilterEnv(obs, raw, penalty)])

    tb_log = f"{tensorboard_logs_base_dir.rstrip('/')}/{pair}/{timeframe}/{datetime.utcnow():%Y%m%d%H%M%S}"
    total_steps = int(hp.get("total_timesteps", 500_000))

    ppo_cfg = hp.get("ppo_hyperparameters", {})
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
        seed=SEED,
        verbose=1,
        tensorboard_log=tb_log,
    )

    logger.info("🏋️  Entrenando PPO (%d timesteps)…", total_steps)
    model.learn(total_timesteps=total_steps, progress_bar=True)
    logger.info("Entrenamiento completado ✔️")

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    final_uri = (
        f"{output_gcs_base_dir.rstrip('/')}/{pair}/{timeframe}/"
        f"{timestamp}/ppo_filter_model.zip"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        local_zip = Path(tmpdir, "model.zip")
        model.save(local_zip)
        gcs_utils.upload_gcs_file(local_zip, final_uri)

    logger.info("Modelo PPO subido a %s", final_uri)
    gc.collect()
    return final_uri


# ──────────────────────────── CLI / Entrypoint ────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Train RL PPO filter")
    p.add_argument("--params-path", required=True)
    p.add_argument("--rl-data-path", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--output-gcs-base-dir", default=constants.RL_MODELS_PATH)
    p.add_argument("--tensorboard-logs-base-dir", default=constants.TENSORBOARD_LOGS_PATH)
    args = p.parse_args()

    final_path = run_rl_training(
        params_path=args.params_path,
        rl_data_path=args.rl_data_path,
        pair=args.pair,
        timeframe=args.timeframe,
        output_gcs_base_dir=args.output_gcs_base_dir,
        tensorboard_logs_base_dir=args.tensorboard_logs_base_dir,
    )

    Path("/tmp/trained_rl_model.txt").write_text(final_path)
    print(f"Trained RL model stored at: {final_path}")
