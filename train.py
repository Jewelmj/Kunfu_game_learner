import torch
import numpy as np
import random
import time
from collections import deque

from config import *
from utils.wrapper import make_env_agent
from agents.dqn_agent import DQNAgent


def linear_schedule(start_e, end_e, decay_steps, current_step):
    """
    Linear decay for epsilon-greedy exploration rate.
    """
    if current_step < decay_steps:
        return start_e - (start_e - end_e) * (current_step / decay_steps)
    else:
        return end_e

def train_agent():
    env = make_env_agent(ENV_ID, frame_stack_k=NUM_FRAMES)
    
    action_size = env.action_space.n
    agent = DQNAgent(action_size, BUFFER_SIZE, LEARNING_RATE)
    
    print(f"Agent initialized. Total Actions: {action_size}")
    
    current_step = 0
    episodes = 0
    
    # Keep track of the last 100 episode rewards for monitoring
    scores_window = deque(maxlen=100)
    best_avg_score = -np.inf
    
    state, info = env.reset(seed=42)

    try:
        start_time = time.time()
        
        while current_step < TOTAL_TIMESTEPS:
            epsilon = linear_schedule(EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS, current_step)
            action = agent.act(state, epsilon)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.save_step(state, action, reward, next_state, done)
            
            state = next_state
            current_step += 1
            
            if current_step > TRAINING_STARTS:
                loss = agent.learn()
                
                if current_step % 10000 == 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = current_step / elapsed_time
                    print(f"Step: {current_step}/{TOTAL_TIMESTEPS} | Epsilon: {epsilon:.3f} | Loss: {loss:.4f} | Speed: {steps_per_sec:.2f} steps/s")

            if done:
                episodes += 1

                episode_reward = info.get('episode_reward', 0) 
                scores_window.append(episode_reward)
                
                # Reset environment for the next episode
                state, info = env.reset()

                # --- Evaluation and Checkpoint Saving ---
                if episodes % 10 == 0:
                    avg_score = np.mean(scores_window)
                    max_score = np.max(scores_window) 
                    
                    print(f"EP {episodes:<5} | Avg Score (100): {avg_score:.2f} | Max Score (100): {max_score:.1f} | Current Score: {episode_reward:.1f}")
                    
                    if avg_score > best_avg_score:
                        best_avg_score = avg_score
                        save_path = f'saved_models/best_kungfu_dqn_{int(avg_score)}.pth'
                        agent.q_network.save_checkpoint(save_path)
                        print(f"*** NEW BEST MODEL SAVED *** (Avg Score: {best_avg_score:.2f})")

        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final model...")
        
    finally:
        final_save_path = './saved_models/kungfu_dqn_final.pth'
        agent.q_network.save_checkpoint(final_save_path)
        print(f"\nFinal model saved to: {final_save_path}")
        env.close()

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    train_agent()