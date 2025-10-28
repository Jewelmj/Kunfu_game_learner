import torch

# Base Enviorment
ENV_ID = "ALE/KungFuMaster-v5"
RENDER_MODE_AGENT = "rgb_array"
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_STACK_K = 2
FRAME_SKIP = 2

# base game ui | for human 
RENDER_MODE_HUMAN = "human"
STEP_DELAY = 0.05
DEFAULT_ACTION = 3

# Agent selection
AGENT_TYPE = 'DQN'    # Options: 'DQN', 'PPO', 'LINEAR'
N_ENVS = 10            # Number of parallel environments

# --- Epsilon-Greedy ---
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 1_000_000

# Off policy learning hyper parameters
BUFFER_SIZE = 50000
BATCH_SIZE = 100
GAMMA = 0.99
LEARNING_RATE = 0.0000625
TARGET_UPDATE_FREQ = 5_000
TRAINING_STARTS = 1000
TOTAL_TIMESTEPS = 100_000

NUM_FRAMES = 4  # Stacked frames for temporal context
CONV_OUT_CHANNELS = [32, 64, 64]
CONV_KERNELS = [8, 4, 3]
CONV_STRIDES = [4, 2, 1]
FC_UNITS1 = 512
FC_UNITS2 = 512
CONV_OUT_SIZE = CONV_OUT_CHANNELS[2] * (
    (((((FRAME_HEIGHT - CONV_KERNELS[0]) // CONV_STRIDES[0] + 1)
       - CONV_KERNELS[1]) // CONV_STRIDES[1] + 1
       - CONV_KERNELS[2]) // CONV_STRIDES[2] + 1) ** 2
)

# Onpolicy hyper parameters
PPO_ROLLOUT_STEPS = 2048       # Number of steps before each PPO update
PPO_EPOCHS = 4
PPO_CLIP_EPSILON = 0.1
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.01
PPO_LR = 0.00025
PPO_BATCH_SIZE = 64
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95

# Log file
LOG_FOLDER = "results"
LOG_FILE = f"{LOG_FOLDER}/log/training_log_{AGENT_TYPE}.csv"
LOG_EVERY_N_STEPS = 300
LOG_CURRENT_BEST_MODEL_AS = f"{LOG_FOLDER}/saved_models/best_kungfu_{AGENT_TYPE}_"
LOG_FINAL_BEST_MODEL_AS = f"{LOG_FOLDER}/saved_models/kungfu_{AGENT_TYPE}_final"

# video file location
MODEL_NAME_FOR_VIDEO = f"saved_models/kungfu_{AGENT_TYPE}_final"
OUTPUT_VIDEO_FILE_NAME = f"video/evaluation_video{AGENT_TYPE}"

# plot log files
LOG_FILES_TO_PLOT = [f"results/log/training_log_{AGENT_TYPE}.csv",
                    f"results/log/training_log_{AGENT_TYPE}.csv"]

# initialise device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
