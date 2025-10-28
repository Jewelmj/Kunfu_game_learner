import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from models.cnn_policy_network import ActorCriticNetwork
from config import DEVICE, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, PPO_EPOCHS, PPO_CLIP_EPSILON, PPO_VF_COEF, PPO_ENT_COEF, PPO_LR
from memory.rollout_buffer import RolloutBuffer

class PPOAgent:
    """
    Proximal Policy Optimization Agent. On-policy, collects data in a RolloutBuffer.
    """
    def __init__(self, action_size, rollout_size, lr=PPO_LR):
        self.action_size = action_size
        self.net = ActorCriticNetwork(action_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        state_shape = (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH) 
        self.memory = RolloutBuffer(rollout_size, state_shape)
        
        print(f"PPO Agent initialized. Rollout Buffer Size: {rollout_size}")

    def get_action_and_value(self, state):
        """
        Gets action, log probability, and state value for the given state(s).
        Handles both single and batch states.
        """
        is_single_state = (state.ndim == 3)
        
        if is_single_state:
            state_tensor = torch.from_numpy(state[np.newaxis, ...]).float().to(DEVICE)
        else:
            state_tensor = torch.from_numpy(state).float().to(DEVICE)
            
        self.net.eval()
        with torch.no_grad():
            action_logits, value = self.net(state_tensor)
        self.net.train()
        
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        action_np = action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        value_np = value.squeeze(-1).cpu().numpy()
        
        if is_single_state:
            return action_np[0].item(), log_prob_np[0].item(), value_np[0].item()
        else:
            return action_np, log_prob_np, value_np

    def save_step(self, state, action, reward, done, log_prob, value):
        self.memory.add(state, action, reward, done, log_prob, value)

    def learn(self, last_value):
        """
        Performs the PPO optimization loop over the collected trajectory.
        `last_value` is the predicted value V(s_T) of the final state in the trajectory.
        """
        self.memory.compute_returns_and_advantages(last_value)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        num_minibatches = 0
        
        for _ in range(PPO_EPOCHS):
            for states, actions, returns, advantages, old_log_probs in self.memory.sample_minibatches():
                new_logits, new_values = self.net(states)
                dist = Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns)
                entropy_loss = -dist.entropy().mean()
                total_loss = policy_loss + PPO_VF_COEF * value_loss + PPO_ENT_COEF * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_minibatches += 1

        self.memory.clear()

        if num_minibatches > 0:
            avg_loss = (total_policy_loss + total_value_loss) / num_minibatches
            return avg_loss
        else:
            return 0.0