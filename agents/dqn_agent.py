import torch
import torch.nn.functional as F
import numpy as np
import random
from models.q_network import QNetwork
from memory.replay_buffer import ReplayBuffer
from config import GAMMA, BATCH_SIZE, LEARNING_RATE, DEVICE, TARGET_UPDATE_FREQ

class DQNAgent:
    def __init__(self, action_size, buffer_size, lr):
        
        self.action_size = action_size
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.update_count = 0

        # --- Networks ---
        self.q_network = QNetwork(action_size).to(DEVICE)
        self.target_network = QNetwork(action_size).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def act(self, state, epsilon):
        """
        Performs an epsilon-greedy action selection.
        """
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).float().to(DEVICE)
            state = state.unsqueeze(0) 
            
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            
            return action_values.argmax().item()

    def save_step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        """
        Performs one optimization step using a batch from the replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(DEVICE)
        actions = torch.from_numpy(actions).long().to(DEVICE)
        rewards = torch.from_numpy(rewards).float().to(DEVICE)
        next_states = torch.from_numpy(next_states).float().to(DEVICE)
        dones = torch.from_numpy(dones).float().to(DEVICE)

        # select the Q-value corresponding to the taken action
        q_currents = self.q_network(states).gather(1, actions.unsqueeze(-1))

        with torch.no_grad():
            q_targets_next = self.target_network(next_states).max(1)[0].unsqueeze(-1)
        
        # TD Target
        q_targets = rewards.unsqueeze(-1) + (self.gamma * q_targets_next * (1 - dones.unsqueeze(-1)))

        loss = F.mse_loss(q_currents, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_count += 1
        if self.update_count % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
            
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
