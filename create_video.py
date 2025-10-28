import os
import torch
import argparse

from config import *
from utils.wrapper import _make_single_env, HighResRenderWrapper
from utils.video_util import show_video_of_model
from utils.agent_loader import get_agent_class

def evaluate_agent(model_path):
    if model_path is None:
        model_file = f"{MODEL_NAME_FOR_VIDEO}.pth"
        model_path = os.path.join(LOG_FOLDER, model_file)
        print(f"No model path specified. Loading default final model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please check the LOG_FOLDER and MODEL_NAME_FOR_VIDEO variables in config.py, or run training first.")
        return

    temp_env = _make_single_env() 
    action_size = temp_env.action_space.n 
    temp_env.close()

    if AGENT_TYPE=="PPO":
        AgentClass = get_agent_class(AGENT_TYPE, True)
    else:
        AgentClass = get_agent_class(AGENT_TYPE, False)
    agent = AgentClass(action_size, buffer_size=1000, lr=0.0)  # irrelevent 2 number.
    
    if not hasattr(agent, 'q_network'):
        print(f"Error: Agent {AGENT_TYPE} does not have a 'q_network' attribute to load.")
        return
        
    try:
        agent.q_network.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.q_network.eval() 
        print(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    print("Starting high-resolution video generation...")
    
    def make_video_env_factory():
        agent_env = _make_single_env()
        return HighResRenderWrapper(agent_env)

    show_video_of_model(agent, make_video_env_factory, output_filename=f"{LOG_FOLDER}/{OUTPUT_VIDEO_FILE_NAME}.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent and generate a video.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
        help=f"Path to the saved model .pth file. If not provided, will load the final model from '{{LOG_FOLDER}}/{{MODEL_NAME_FOR_VIDEO}}.pth'"
    )
    args = parser.parse_args()
    
    evaluate_agent(args.model_path)
