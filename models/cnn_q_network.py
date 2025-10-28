import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_FRAMES, CONV_OUT_CHANNELS, CONV_KERNELS, CONV_STRIDES, FC_UNITS1, FC_UNITS2, CONV_OUT_SIZE

class QNetwork(nn.Module):
    def __init__(self, action_size):
        """
        Initializes the Q-Network.
        Input: (NUM_FRAMES, 84, 84)
        Output: Q-values for each possible action.
        """
        super(QNetwork, self).__init__()

        # 1. Input: (NUM_FRAMES, 84, 84) -> Output: (32, 20, 20)
        self.conv1 = nn.Conv2d(in_channels=NUM_FRAMES, out_channels=CONV_OUT_CHANNELS[0], kernel_size=CONV_KERNELS[0], stride=CONV_STRIDES[0])
        self.conv2 = nn.Conv2d(in_channels=CONV_OUT_CHANNELS[0], out_channels=CONV_OUT_CHANNELS[1], kernel_size=CONV_KERNELS[1], stride=CONV_STRIDES[1])
        self.conv3 = nn.Conv2d(in_channels=CONV_OUT_CHANNELS[1], out_channels=CONV_OUT_CHANNELS[2], kernel_size=CONV_KERNELS[2], stride=CONV_STRIDES[2]) # 64, 9, 9) -> (64, 7, 7)
        
        # 4. Input: 3136 -> Output: 512
        self.fc4 = nn.Linear(CONV_OUT_SIZE, FC_UNITS1)
        self.fc5 = nn.Linear(FC_UNITS1, FC_UNITS2)
        self.fc6 = nn.Linear(FC_UNITS2, action_size)

    def forward(self, x):
        """
        Forward pass through the network.
        Input x should be normalized (0.0-1.0) and on the correct device.
        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) # Flattens all dimensions except batch size

        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
