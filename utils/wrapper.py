import gymnasium as gym
import ale_py # for gym to recoganise our game.
import cv2

cv2.ocl.setUseOpenCL(False)

def make_env_human(env_id):
    return gym.make(env_id, render_mode="human")

def make_env_agent(env_id, frame_stack_k=4):
    env = gym.make(env_id, render_mode='rgb_array', frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=True,
    )
    env = gym.wrappers.FrameStackObservation(env, frame_stack_k)
    return env