import torch
import numpy as np
import random
import tqdm
import csv
from collections import deque

from config import *
from utils.wrapper import make_parallel_env 
from utils.agent_loader import get_agent_class

def linear_schedule(start_e, end_e, decay_steps, current_step):
    """
    Linear decay for epsilon-greedy exploration rate.
    """
    if current_step < decay_steps:
        return start_e - (start_e - end_e) * (current_step / decay_steps)
    else:
        return end_e

def train_agent():
    envs = make_parallel_env(N_ENVS)

    action_size = envs.single_action_space.n
    
    AgentClass = get_agent_class(AGENT_TYPE, False)
    agent = AgentClass(action_size=len(ALLOWED_ACTIONS), rollout_size=PPO_ROLLOUT_STEPS * N_ENVS)
    
    print(f"Agent initialized: {AGENT_TYPE}. Total Actions: {action_size}")
    print(f"Using {N_ENVS} parallel environments for faster data collection.")
    
    episodes = 0
    loss = 0.0 

    scores_window = deque(maxlen=100)
    best_avg_score = -np.inf
    
    state, info = envs.reset(seed=42)
    episode_rewards = np.zeros(N_ENVS, dtype=np.float32)

    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "epsilon", "avg_reward", "max_reward", "loss"])

    last_avg_score, last_max_score, last_episodes = 0.0, 0.0, 0
    last_epsilon, last_loss = 0.0, 0.0

    try:
        with tqdm.tqdm(total=TOTAL_TIMESTEPS, desc="Training Progress", unit="step") as pbar:
            current_step = 0
            while current_step < TOTAL_TIMESTEPS:
                epsilon = linear_schedule(EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS, current_step)
                actions = agent.act(state, epsilon)
                
                next_state, reward, terminated, truncated, info = envs.step(actions)
                dones = terminated | truncated
                episode_rewards += reward

                for i in range(N_ENVS):
                    agent.save_step(state[i], actions[i], reward[i], next_state[i], dones[i])
                
                state = next_state
                current_step += N_ENVS
                pbar.update(N_ENVS)
                
                if current_step >= TRAINING_STARTS:
                    loss = agent.learn()
                    
                
                # --- Handle Episode Termination and Logging ---
                finished_rewards = []
                
                for i, done in enumerate(dones):
                    if done:
                        episodes += 1
                        finished_rewards.append(episode_rewards[i])
                        episode_rewards[i] = 0 

                if finished_rewards:
                    scores_window.extend(finished_rewards)
                    
                    last_avg_score = np.mean(scores_window)
                    last_max_score = np.max(scores_window)
                    last_episodes = episodes
                    last_epsilon = epsilon
                    last_loss = loss

                    if last_avg_score > best_avg_score:
                        best_avg_score = last_avg_score
                        save_path = f"{LOG_CURRENT_BEST_MODEL_AS}{int(last_avg_score)}.pth"
                        agent.q_network.save_checkpoint(save_path)

                pbar.set_postfix(
                    eps=f"{last_epsilon:.3f}",
                    loss=f"{last_loss:.4f}",
                    avgR=f"{last_avg_score:.1f}",
                    maxR=f"{last_max_score:.0f}",
                    ep=f"{last_episodes}",
                    refresh=False
                )

                if current_step % LOG_EVERY_N_STEPS == 0:
                    with open(LOG_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            current_step,
                            last_episodes,
                            round(last_epsilon, 5),
                            round(last_avg_score, 5),
                            round(last_max_score, 5),
                            round(last_loss, 5)
                        ])

        
    except KeyboardInterrupt:
        tqdm.tqdm.write("\nTraining interrupted by user. Saving final model...")
        
    finally:
        final_save_path = f'./{LOG_FINAL_BEST_MODEL_AS}.pth'
        agent.q_network.save_checkpoint(final_save_path)
        tqdm.tqdm.write(f"\nFinal model saved to: {final_save_path}")
        envs.close() 

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
