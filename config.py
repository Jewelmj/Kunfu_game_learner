import os
from dotenv import load_dotenv
import torch

load_dotenv()

# --- Environment Settings ---
ENV_ID = os.getenv('env_id', "ALE/KungFuMaster-v5")
NUM_FRAMES = os.getenv('num_frames', 4)           # Number of frames to stack (for motion detection)
FRAME_WIDTH = os.getenv('frame_width', 84)           # Resized image width
FRAME_HEIGHT = os.getenv('frame_height', 84)         # Resized image height
RENDER_MODE_AGENT = os.getenv('render_mode_agent', "rgb_array")

# --- Frame stack and skip ---
FRAME_STACK_K = int(os.getenv('frame_stack_k', 4))    # Number of frames to stack for RL agent
FRAME_SKIP = int(os.getenv('frame_skip', 4))          # Number of frames to skip per action

# --- base game ui | for human ---
RENDER_MODE_HUMAN = os.getenv('render_mode_human', "human")
STEP_DELAY = os.getenv('step_delay', 0.05)
DEFAULT_ACTION = os.getenv('default_action', 3)
