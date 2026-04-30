"""
train.py — Train a PPO agent with curriculum learning.
Usage:  python train.py
        python train.py --steps 1000000
"""
import os, sys, argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parking_env import ParkingEnv as _ParkingEnv, NUM_RAYS, CURRICULUM

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_3")
LOGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "model_3")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


class ParkingGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        import pygame; pygame.init()
        self._env = _ParkingEnv.__new__(_ParkingEnv)
        self._env._build_world()
        self._env.agent           = None
        self._env.steps           = 0
        self._env.done            = False
        self._env.collision       = False
        self._env.status          = ""
        self._env.reward_last     = 0.0
        self._env.prev_dist       = 0.0
        self._env.curriculum_stage= 0
        self._env.episode_results = []
        self._env.window_size     = 20

        obs_len = 11 + NUM_RAYS
        self.observation_space = spaces.Box(-2.0, 2.0, (obs_len,), np.float32)
        self.action_space      = spaces.Box(-1.0,  1.0, (3,),      np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        acc   = float(action[0])
        steer = float(action[1])
        gear  = -1 if float(action[2]) < 0 else 1
        obs, reward, done = self._env.step((max(0,acc), max(0,-acc), steer, gear))
        return obs, float(reward), done, False, {}

    def render(self): pass
    def close(self):  pass


class CurriculumLogCallback(BaseCallback):
    """Prints curriculum stage changes during training."""
    def __init__(self, env):
        super().__init__()
        self._park_env = env
        self._last_stage = 0

    def _on_step(self):
        stage = self._park_env._env.curriculum_stage
        if stage != self._last_stage:
            self._last_stage = stage
            print(f"\n[CURRICULUM] ── Now on {CURRICULUM[stage]['name']} ──\n")
        return True


def train(total_steps=500_000, resume_path=None):
    env = ParkingGymEnv()

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=MODELS_DIR,
        name_prefix="ppo_parking",
        verbose=1,
    )
    curriculum_cb = CurriculumLogCallback(env)

    if resume_path:
        print(f"[TRAIN] Resuming from {resume_path}")
        model = PPO.load(resume_path, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=2e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,       # encourage exploration
            device="cpu",        # MLP policy is faster on CPU
            verbose=1,
            tensorboard_log=LOGS_DIR,
        )

    print(f"[TRAIN] Training for {total_steps:,} steps  |  checkpoints → {MODELS_DIR}")
    model.learn(total_timesteps=total_steps,
                callback=[checkpoint_cb, curriculum_cb],
                reset_num_timesteps=resume_path is None)

    final = os.path.join(MODELS_DIR, "ppo_parking_final")
    model.save(final)
    print(f"[TRAIN] Done. Saved → {final}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=1_000_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .zip model to continue training")
    args = parser.parse_args()
    train(args.steps, args.resume)