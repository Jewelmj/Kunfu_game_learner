import torch
import numpy as np

from config import DEVICE, PPO_BATCH_SIZE, PPO_GAMMA, PPO_GAE_LAMBDA

class RolloutBuffer:
    """
    Stores a full trajectory (rollout) for on-policy learning.
    It clears after every major learning step.
    """
    def __init__(self, buffer_size, state_shape):
        self.states = np.empty((buffer_size, *state_shape), dtype=np.uint8)
        self.actions = np.empty(buffer_size, dtype=np.int32)  
        self.rewards = np.empty(buffer_size, dtype=np.float32)  
        self.dones = np.empty(buffer_size, dtype=np.bool_)
        self.log_probs = np.empty(buffer_size, dtype=np.float32)  
        self.values = np.empty(buffer_size, dtype=np.float32)  
        self.returns = np.empty(buffer_size, dtype=np.float32) 
        self.advantages = np.empty(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.max_size = buffer_size

    def add(self, state, action, reward, done, log_prob, value):
        if self.ptr >= self.max_size:
            raise IndexError("Rollout buffer exceeded max capacity.")
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value

        self.ptr += 1

    def clear(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, last_values):
        """
        Computes the Generalised Advantage Estimate (GAE) and Returns
        for the collected trajectory.
        
        last_values: array of shape (N_ENVS,) containing V(s_T) for each environment
        """
        if np.any(np.isnan(last_values)):
            print("WARNING: last_values contains NaN!")
            last_values = np.nan_to_num(last_values, nan=0.0)
        
        if np.isscalar(last_values):
            last_values = np.array([last_values])
        
        n_envs = len(last_values)
        steps_per_env = self.ptr // n_envs
        
        rewards = self.rewards[:self.ptr].reshape(steps_per_env, n_envs)
        dones = self.dones[:self.ptr].reshape(steps_per_env, n_envs)
        values = self.values[:self.ptr].reshape(steps_per_env, n_envs)
        
        advantages = np.zeros((steps_per_env, n_envs), dtype=np.float32)
        last_gae_lam = np.zeros(n_envs, dtype=np.float32)
        
        for t in reversed(range(steps_per_env)):
            if t == steps_per_env - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + PPO_GAMMA * next_values * next_non_terminal - values[t]
            
            advantages[t] = last_gae_lam = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_non_terminal * last_gae_lam
        
        self.advantages[:self.ptr] = advantages.flatten()
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
        
        self.advantages[:self.ptr] = (self.advantages[:self.ptr] - self.advantages[:self.ptr].mean()) / (self.advantages[:self.ptr].std() + 1e-8)

    def sample_minibatches(self):
        """Yields minibatches for PPO optimization."""
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        
        for start in range(0, self.ptr, PPO_BATCH_SIZE):
            end = start + PPO_BATCH_SIZE
            batch_indices = indices[start:end]

            yield (
                torch.from_numpy(self.states[batch_indices]).float().to(DEVICE),
                torch.from_numpy(self.actions[batch_indices]).long().to(DEVICE).unsqueeze(-1),
                torch.from_numpy(self.returns[batch_indices]).float().to(DEVICE).unsqueeze(-1),
                torch.from_numpy(self.advantages[batch_indices]).float().to(DEVICE).unsqueeze(-1),
                torch.from_numpy(self.log_probs[batch_indices]).float().to(DEVICE).unsqueeze(-1),
                torch.from_numpy(self.values[batch_indices]).float().to(DEVICE).unsqueeze(-1)
            )