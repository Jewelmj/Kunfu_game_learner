import numpy as np
from config import FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES

class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Initializes the replay buffer.
        State shape is (NUM_FRAMES, HEIGHT, WIDTH)
        """
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

        self.states = np.empty((buffer_size, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        self.actions = np.empty(buffer_size, dtype=np.int64)
        self.rewards = np.empty(buffer_size, dtype=np.float32)
        self.next_states = np.empty((buffer_size, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        self.dones = np.empty(buffer_size, dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        """
        Adds a transition to the buffer.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Advance the pointer, wrapping around if buffer_size is reached
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        """
        Samples a random batch of transitions for training.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        # Normalize pixel values to float (0.0-1.0) for the network
        states = states.astype(np.float32) / 255.0
        next_states = next_states.astype(np.float32) / 255.0

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current number of stored transitions.
        """
        return self.size
