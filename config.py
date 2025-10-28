import torch

# --- Environment Settings ---
ENV_ID = "ALE/KungFuMaster-v5"
N_ENVS = 12                # number of parelel enviorments
RENDER_MODE_AGENT = "rgb_array"

# --- image preprocess in utils/wrapper ---
FRAME_WIDTH = 84   
FRAME_HEIGHT = 84  
FRAME_STACK_K = 2  
FRAME_SKIP = 2       

# --- base game ui | for human ---
RENDER_MODE_HUMAN = "human"
STEP_DELAY = 0.05
DEFAULT_ACTION = 3

# --- Agent select ---
AGENT_TYPE = 'DQN' # Options: 'DQN', 'PPO', 'LINEAR'

# --- CNN Hyperparameters ---
NUM_FRAMES = 4          # for motion detection.
CONV_OUT_CHANNELS = [32, 64, 64]
CONV_KERNELS = [8, 4, 3]
CONV_STRIDEs = [4, 2, 1]
FC_UNITS1 = 512   # common for DQN, linier, and PPO model.
FC_UNITS2 = 512   # common for DQN, linier, and PPO model.
CONV_OUT_SIZE = CONV_OUT_CHANNELS[2] * \
                (((((FRAME_HEIGHT - CONV_KERNELS[0]) // CONV_STRIDEs[0] + 1) - CONV_KERNELS[1]) // CONV_STRIDEs[1] + 1
                  - CONV_KERNELS[2]) // CONV_STRIDEs[2] + 1) ** 2

# --- DQN Hyperparameters ---
BUFFER_SIZE = 50000
BATCH_SIZE = 100               # Replay buffer sampling
GAMMA = 0.99                  # Discount factor
LEARNING_RATE = 0.0000625
TARGET_UPDATE_FREQ = 10000    # Frequency (in steps) of updating the target network
TRAINING_STARTS = 1000        # training begins at
TOTAL_TIMESTEPS = 1000000 

# --- Epsilon-Greedy Schedule ---
EPSILON_START = 1.0          
EPSILON_END = 0.1            
EPSILON_DECAY_STEPS = 1000000 

# --- LOG ---
LOG_FOLDER = "results"
LOG_FILE = f"{LOG_FOLDER}/log/training_log.csv"
LOG_EVERY_N_STEPS = 1000
LOG_CURRENT_BEST_MODEL_AS = f"{LOG_FOLDER}/saved_models/best_kungfu_{AGENT_TYPE}_"
LOG_FINAL_BEST_MODEL_AS = f"{LOG_FOLDER}/saved_models/kungfu_{AGENT_TYPE}_final"

# --- Video ---
MODEL_NAME_FOR_VIDEO = f"saved_models/kungfu_{AGENT_TYPE}_final"
OUTPUT_VIDEO_FILE_NAME = "video/evaluation_video"

# --- Hardware Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
