"""
Author: Minh Pham-Dinh
Created: Jan 7th, 2024
Last Modified: Jan 10th, 2024
Email: mhpham26@colby.edu

Description:
    Utilities file for used with RL files
"""

import numpy as np
import torch
from collections import deque
from torch.utils.data import Dataset

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


# Prioritized Experience Replay
class PrioritizedReplayBuffer:
    def __init__(
            self,
            random,
            buffer_size: int = 10000,
            batch_size: int = 64,
            alpha: float = 0):
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

        indices = self.random.choice(
            len(self.buffer), self.batch_size, p=prob_weights)
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


class PPOMemoryMultis(Dataset):
    def __init__(self, memory_size, obs_shape, num_envs, device, continuous=False, act_shape=None):
        self.total_length = memory_size * num_envs
        self.obs_shape = obs_shape
        self.states = torch.zeros((memory_size, num_envs) + obs_shape, dtype=torch.float, device=device)
        if continuous:
            if act_shape is None:
                raise Exception('please provide shape of action space for continuous problem')
            self.actions = torch.zeros((memory_size, num_envs) + act_shape, dtype=torch.float, device=device)
        else:
            self.actions = torch.zeros((memory_size, num_envs), dtype=torch.long, device=device)
        self.probs = torch.zeros((memory_size, num_envs), dtype=torch.float, device=device)
        self.rewards = torch.zeros((memory_size, num_envs), dtype=torch.float, device=device)
        self.dones = torch.zeros((memory_size, num_envs), dtype=torch.float, device=device)
        self.values = torch.zeros((memory_size, num_envs), dtype=torch.float, device=device)
        self.pointer = 0
        
        print(f'''
-----------initialized memory----------              

states_buffer_shape: {self.states.shape}
actions_buffer_shape: {self.actions.shape}
log_probs_buffer_shape: {self.probs.shape}
rewards_buffer_shape: {self.rewards.shape}
dones_buffer_shape: {self.dones.shape}

----------------------------------------
              ''')

    def add(self, state, action, log_prob, reward, done, value):
        if self.pointer >= len(self.actions):
            raise Exception('Max memory exceeded')
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.probs[self.pointer] = log_prob
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.values[self.pointer] = value
        self.pointer += 1

    def canOptimize(self):
        return self.pointer == self.states.shape[0]

    def clear(self):
        self.pointer = 0
    
    def get_data(self):
        return self.states, self.actions, self.rewards, self.probs, self.dones, self.values

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if self.pointer < len(self.states):
            raise Exception('not ready to sample')
        return idx
    

class SACMemory():
    def __init__(self, memory_size, env, random):
        """SAC memory - a replay buffer

        Args:
            memory_size (int): max memory size before
            env (_type_): gymnasium environment
            random (np.Random): numpy random generator
            device (torch.device): device to store on 'cpu', 'cuda' or 'mps'
        """
        self.total_length = memory_size
        self.random = random
        self.states = torch.zeros((memory_size, )+ env.observation_space.shape, dtype=torch.float)
        self.actions = torch.zeros((memory_size, ) + env.action_space.shape, dtype=torch.float)
        self.rewards = torch.zeros((memory_size, ), dtype=torch.float)
        self.next_states = torch.zeros((memory_size, ) + env.observation_space.shape, dtype=torch.float)
        self.dones = torch.zeros((memory_size, ), dtype=torch.float)
        self.filled = False
        self.pointer = 0
        
        print(f'''
-----------initialized memory----------              

states_buffer_shape: {self.states.shape}
actions_buffer_shape: {self.actions.shape}
rewards_buffer_shape: {self.rewards.shape}
next_states_buffer_shape: {self.next_states.shape}
dones_buffer_shape: {self.dones.shape}

----------------------------------------
              ''')

    def add(self, state, action, reward, next_state, done):
        if self.pointer >= self.states.shape[0]:
            if not self.filled:
                self.filled = True
            self.pointer = 0
        
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        self.pointer += 1
    
    def to_device(self, device):
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.dones = self.dones.to(device)
    
    def sample(self, batch_size):
        if len(self) < batch_size and not self.filled:
            raise Exception('not enough data to start sample, please lower minibatch size or collect more experiences')
        
        idxs = self.random.choice(
            np.arange(0, len(self), 1), batch_size)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]

    def __len__(self):
        return self.pointer