import numpy as np
import torch
from collections import deque

def argmax(arr, random):
    max_val = float('-inf')
    ties = []
    for i in range(arr.shape[0]):
        if arr[i] > max_val:
            max_val = arr[i]
            ties = [i]
        elif arr[i] == max_val:
            ties.append(i)
    return random.choice(ties)    

def parse_hyperparameters(file_path):
    hyperparameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                values = [float(val.strip()) for val in parts[1].split(',')]
                hyperparameters[key] = values
    return hyperparameters

class ReplayBuffer():
    def __init__(self, random, buffer_size:int = 10000, batch_size:int = 64):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.random = random
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        idxs = self.random.choice(np.arange(0, len(self.buffer), 1), self.batch_size)
        experiences = [self.buffer[i] for i in idxs]
        return experiences
        # states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))

        # return torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(next_states), torch.from_numpy(dones)

    def __len__(self):
        return len(self.buffer)


# Prioritized Experience Replay
class PrioritizedReplayBuffer:
    def __init__(self, random, buffer_size:int = 10000, batch_size:int = 64, alpha:float = 0, beta:float = 0):
        self.alpha = alpha  # Priority exponent
        self.capacity = buffer_size
        self.batch_size = batch_size
        
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0
        self.random = random

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, beta):
        priorities = np.array(self.priorities)
        prob_weights = priorities ** self.alpha
        prob_weights /= prob_weights.sum()

        indices = self.random.choice(len(self.buffer), self.batch_size, p=prob_weights)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * prob_weights[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, new_priorities):
        new_priorities = new_priorities.detach().cpu().numpy()
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)     