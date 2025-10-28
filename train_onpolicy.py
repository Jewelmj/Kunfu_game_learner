import torch
import numpy as np
import random
import tqdm
import csv
from collections import deque

from config import DEVICE, N_ENVS, AGENT_TYPE, PPO_ROLLOUT_STEPS, LOG_FILE, TOTAL_TIMESTEPS, LOG_CURRENT_BEST_MODEL_AS, LOG_FINAL_BEST_MODEL_AS, LOG_EVERY_N_STEPS
from utils.wrapper import make_parallel_env
from utils.agent_loader import get_agent_class


def train_ppo():
    envs = make_parallel_env(N_ENVS)
    action_size = envs.single_action_space.n

    AgentClass = get_agent_class(AGENT_TYPE, True)
    agent = AgentClass(action_size, rollout_size=PPO_ROLLOUT_STEPS)
    
    print(f"PPO Training started — using {N_ENVS} parallel environments")
    print(f"Action size: {action_size}, Rollout length: {PPO_ROLLOUT_STEPS}")

    state, info = envs.reset(seed=42)
    scores_window = deque(maxlen=100)
    best_avg_score = -np.inf
    episode_rewards = np.zeros(N_ENVS, dtype=np.float32)
    episodes = 0

    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "avg_reward", "max_reward", "loss"])

    total_steps = 0
    try:
        with tqdm.tqdm(total=TOTAL_TIMESTEPS, desc="PPO Training", unit="step") as pbar:
            while total_steps < TOTAL_TIMESTEPS:
                agent.memory.clear()
                for _ in range(PPO_ROLLOUT_STEPS):
                    actions, log_probs, values = agent.get_action_and_value(state)
                    next_state, reward, terminated, truncated, info = envs.step(actions)
                    dones = terminated | truncated

                    for i in range(N_ENVS):
                        agent.save_step(
                            state[i],
                            actions[i],
                            reward[i],
                            dones[i],
                            log_probs[i],
                            values[i],
                        )

                    episode_rewards += reward
                    state = next_state
                    total_steps += N_ENVS
                    pbar.update(N_ENVS)

                    for i, done in enumerate(dones):
                        if done:
                            scores_window.append(episode_rewards[i])
                            episode_rewards[i] = 0
                            episodes += 1

                with torch.no_grad():
                    _, last_values = agent.net(torch.tensor(state, dtype=torch.float32, device=DEVICE))
                last_values = last_values.squeeze(-1).cpu().numpy()

                avg_loss = agent.learn(last_values.mean())
                avg_reward = np.mean(scores_window) if scores_window else 0.0
                max_reward = np.max(scores_window) if scores_window else 0.0

                if avg_reward > best_avg_score:
                    best_avg_score = avg_reward
                    save_path = f"{LOG_CURRENT_BEST_MODEL_AS}{int(avg_reward)}.pth"
                    agent.net.save_checkpoint(save_path)

                pbar.set_postfix(
                    avgR=f"{avg_reward:.1f}",
                    maxR=f"{max_reward:.0f}",
                    loss=f"{avg_loss:.4f}",
                    ep=f"{episodes}",
                    refresh=False,
                )

                if total_steps % LOG_EVERY_N_STEPS == 0:
                    with open(LOG_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            total_steps,
                            episodes,
                            round(avg_reward, 5),
                            round(max_reward, 5),
                            round(avg_loss, 5)
                        ])

    except KeyboardInterrupt:
        tqdm.tqdm.write("\nTraining interrupted by user. Saving final model...")

    finally:
        final_path = f"./{LOG_FINAL_BEST_MODEL_AS}.pth"
        agent.net.save_checkpoint(final_path)
        tqdm.tqdm.write(f"\nFinal PPO model saved to {final_path}")
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

    train_ppo()