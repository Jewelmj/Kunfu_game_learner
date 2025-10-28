import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT, FC_UNITS1,FC_UNITS2

class LinearQNetwork(nn.Module):
    def __init__(self, action_size):
        super(LinearQNetwork, self).__init__()
        
        input_size = NUM_FRAMES * FRAME_WIDTH * FRAME_HEIGHT
        self.fc1 = nn.Linear(input_size, FC_UNITS1)
        self.fc2 = nn.Linear(FC_UNITS1, FC_UNITS2)
        self.fc3 = nn.Linear(FC_UNITS2, action_size)

    def forward(self, x):
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))