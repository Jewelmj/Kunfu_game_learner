import gymnasium as gym
import ale_py # for gym to recoganise our game.
import numpy as np
import cv2
from config import RENDER_MODE_HUMAN, RENDER_MODE_AGENT, FRAME_STACK_K, FRAME_SKIP, N_ENVS, ENV_ID, NUM_FRAMES

cv2.ocl.setUseOpenCL(False)

def make_env_human(env_id):
    return gym.make(env_id, render_mode=RENDER_MODE_HUMAN)

def _make_training_env(env_id, frame_stack_k=FRAME_STACK_K):
    env = gym.make(env_id, render_mode=RENDER_MODE_AGENT, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=FRAME_SKIP,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=True,
    )
    env = gym.wrappers.FrameStackObservation(env, frame_stack_k)
    return env

def _make_single_env():
    return _make_training_env(ENV_ID, frame_stack_k=NUM_FRAMES)

def make_parallel_env(n_envs=N_ENVS):
    return gym.vector.AsyncVectorEnv([lambda: _make_single_env() for _ in range(n_envs)])