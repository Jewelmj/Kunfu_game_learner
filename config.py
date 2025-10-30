import torch

def _get_conv_out(h_in, k, s):
    return (h_in - k) // s + 1

# Base Enviorment
ENV_ID = "ALE/KungFuMaster-v5"
RENDER_MODE_AGENT = "rgb_array"
ALLOWED_ACTIONS = [0, 1, 2, 3, 4, 7, 8]
FRAME_WIDTH = 90
FRAME_HEIGHT = 90
NUM_FRAMES = 4  # Stacked frames for temporal context
FRAME_SKIP = 4

# base game ui | for human 
RENDER_MODE_HUMAN = "human"
STEP_DELAY = 0.05
DEFAULT_ACTION = 3

# Agent selection
AGENT_TYPE = 'PPO'    # Options: 'DQN', 'PPO', 'LINEAR'
N_ENVS = 10            # Number of parallel environments

# --- Epsilon-Greedy ---
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 10_000_000

# Off policy learning hyper parameters
BUFFER_SIZE = 5000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0000625
TARGET_UPDATE_FREQ = 5_000
TRAINING_STARTS = 1000
TOTAL_TIMESTEPS = 10_000_000

CONV_OUT_CHANNELS = [32, 64, 16, 16]
CONV_KERNELS = [6, 5, 4, 3, 2]
CONV_STRIDES = [3, 2, 1, 1, 1]
FC_UNITS1 = 128
FC_UNITS2 = 128
h1 = _get_conv_out(FRAME_HEIGHT, CONV_KERNELS[0], CONV_STRIDES[0])
h2 = _get_conv_out(h1, CONV_KERNELS[1], CONV_STRIDES[1])
h3 = _get_conv_out(h2, CONV_KERNELS[2], CONV_STRIDES[2])
h4 = _get_conv_out(h3, CONV_KERNELS[3], CONV_STRIDES[3])
h5 = _get_conv_out(h4, CONV_KERNELS[4], CONV_STRIDES[4])

CONV_OUT_SIZE = CONV_OUT_CHANNELS[4] * (h5 ** 2)

# Onpolicy hyper parameters
PPO_ROLLOUT_STEPS = 8000     # Number of steps before each PPO update
PPO_EPOCHS = 5
PPO_LR = 0.0008
PPO_BATCH_SIZE = 1024
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.05

# Log file
LOG_FOLDER = "results"
LOG_FILE = f"{LOG_FOLDER}/log/training_log_{AGENT_TYPE}.csv"
LOG_EVERY_N_STEPS = 500
LOG_CURRENT_BEST_MODEL_AS = f"{LOG_FOLDER}/saved_models/best_kungfu_{AGENT_TYPE}_"
LOG_FINAL_BEST_MODEL_AS = f"{LOG_FOLDER}/saved_models/kungfu_{AGENT_TYPE}_final"

# video file location
MODEL_NAME_FOR_VIDEO = f"saved_models/kungfu_{AGENT_TYPE}_final"
OUTPUT_VIDEO_FILE_NAME = f"video/evaluation_video{AGENT_TYPE}"

# plot settings
LOG_FILES_TO_PLOT = [f"results/log/training_log_{AGENT_TYPE}.csv"]
PLOT_X_MIN = 0              
PLOT_X_MAX = None     

PLOT_REWARD_Y_MIN = 0      
PLOT_REWARD_Y_MAX = None    

PLOT_LOSS_Y_MIN = 0         
PLOT_LOSS_Y_MAX = None     

PLOT_STYLE = 'bmh'  # Options: 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-whitegrid', 'ggplot', 'bmh'
PLOT_FIGSIZE = (14, 7)      
PLOT_DPI = 300              
PLOT_LINEWIDTH = 2.5        
PLOT_ALPHA = 0.85          
PLOT_GRID_ALPHA = 0.3      

PLOT_SMOOTHING_WINDOW = 1 

PLOT_COLORS = [
    '#FF6B6B',  
    '#BB8FCE',  
    '#4ECDC4',  
    '#45B7D1',  
    '#FFA07A',  
    '#98D8C8',  
    '#F7DC6F', 
    '#85C1E2',  
]

# initialise device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
