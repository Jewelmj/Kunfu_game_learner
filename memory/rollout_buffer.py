import torch
import numpy as np

from config import DEVICE, PPO_BATCH_SIZE, PPO_GAMMA, PPO_GAE_LAMBDA

class RolloutBuffer:
    """
    Stores a full trajectory (rollout) for on-policy learning.
    It clears after every major learning step.
    """
    def __init__(self, buffer_size, state_shape):
        self.states = np.empty((buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.empty(buffer_size, dtype=np.int64)
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

    def compute_returns_and_advantages(self, last_value):
        """
        Computes the Generalised Advantage Estimate (GAE) and Returns
        for the collected trajectory.
        """
        last_gae_lam = 0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t+1]
                next_value = self.values[t+1]

            delta = self.rewards[t] + PPO_GAMMA * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_non_terminal * last_gae_lam

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
            )