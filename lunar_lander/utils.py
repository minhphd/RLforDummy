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
        states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))
        
        return torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(next_states), torch.from_numpy(dones)

    def __len__(self):
        return len(self.buffer)
    