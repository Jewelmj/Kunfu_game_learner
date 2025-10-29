import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_FRAMES, FC_UNITS1, FC_UNITS2, CONV_OUT_CHANNELS, CONV_KERNELS, CONV_STRIDES, CONV_OUT_SIZE

class ActorCriticNetwork(nn.Module):
    def __init__(self, action_size):
        """
        Initializes the PPO Actor-Critic Network.
        Shares CNN backbone for feature extraction.
        """
        super(ActorCriticNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=NUM_FRAMES, out_channels=CONV_OUT_CHANNELS[0], kernel_size=CONV_KERNELS[0], stride=CONV_STRIDES[0])
        self.conv2 = nn.Conv2d(in_channels=CONV_OUT_CHANNELS[0], out_channels=CONV_OUT_CHANNELS[1], kernel_size=CONV_KERNELS[1], stride=CONV_STRIDES[1])
        self.conv3 = nn.Conv2d(in_channels=CONV_OUT_CHANNELS[1], out_channels=CONV_OUT_CHANNELS[2], kernel_size=CONV_KERNELS[2], stride=CONV_STRIDES[2]) 
        
        self.actor_fc1 = nn.Linear(CONV_OUT_SIZE, FC_UNITS1)
        self.actor_fc2 = nn.Linear(FC_UNITS1, FC_UNITS2)
        self.actor_head = nn.Linear(FC_UNITS2, action_size) 

        self.critic_fc1 = nn.Linear(CONV_OUT_SIZE, FC_UNITS1)
        self.critic_fc2 = nn.Linear(FC_UNITS1, FC_UNITS2)
        self.critic_head = nn.Linear(FC_UNITS2, 1) 

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        actor_x = F.relu(self.actor_fc1(x))
        actor_x = F.relu(self.actor_fc2(actor_x))
        action_logits = self.actor_head(actor_x)

        # 3. Critic Branch
        critic_x = F.relu(self.critic_fc1(x))
        critic_x = F.relu(self.critic_fc2(critic_x))
        value_estimate = self.critic_head(critic_x)

        return action_logits, value_estimate
    
    def save_checkpoint(self, path):
        """Saves the network weights."""
        torch.save(self.state_dict(), path)
        
    def load_checkpoint(self, path):
        """Loads the network weights."""
        self.load_state_dict(torch.load(path))