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

class HighResRenderWrapper(gym.Wrapper):
    """
    Wraps the fully preprocessed (stacked, grayscale, 84x84) environment.
    
    It keeps the reset/step methods intact (returning the low-res, stacked observation 
    the agent expects), but overrides render() to return the raw, high-resolution, 
    color RGB frame from the base Atari environment for video recording purposes.
    """
    def __init__(self, env):
        super().__init__(env)
        self.base_env = self.env.unwrapped

        if self.base_env.render_mode != 'rgb_array':
            print(f"Warning: Base environment render_mode is '{self.base_env.render_mode}'. Must be 'rgb_array' for high-res output.")  
        print(f"Video Wrapper initialized. Base environment type: {type(self.base_env).__name__}")
        
    def render(self):
        """
        Forces the underlying base environment to render the full-resolution RGB array.
        """
        raw_frame = self.base_env.render()
        
        if raw_frame is not None and len(raw_frame.shape) == 3 and raw_frame.shape[2] == 3:
            return raw_frame
        else:
            print("Error: Could not retrieve a valid high-resolution RGB frame. Returning the default frame.")
            return self.env.render()