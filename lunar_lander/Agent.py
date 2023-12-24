import numpy as np
import gymnasium
import torch
import copy

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


class ReplayBuffer():
    def __init__(self, random: np.random.Generator, buffer_size:int = 10000, batch_size:int = 10000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.random = random
    
    def add(self, payload):
        self.buffer.append(payload)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def sample(self):
        idxs = self.random.choice(np.arange(0, self.size(), 1, dtype=int), size=self.batch_size)
        return [self.buffer[i] for i in idxs]
        
    def size(self):
        return len(self.buffer)
    
    def canSample(self):
        return self.size() >= self.batch_size
    
    def __repr__(self):
        return self.buffer

class Agent():
    def __init__(self, writer, device, env: gymnasium.Env, generator :np.random.Generator, net: torch.nn.Module, eps_start :float = 0.9, eps_end :float = 0.05, lr: float = 0.01, discount :float =0.01,buffer_size:int = 10000, batch_size:int = 10000) -> None:
        self.env = env
        self.target_net = net
        self.predict_net = copy.deepcopy(net)
        self.optimizer = torch.optim.Adam(self.predict_net.parameters(), lr=lr)
        self.experiences = ReplayBuffer(random=generator, buffer_size=buffer_size, batch_size=batch_size)
        self.writer = writer
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        # self.step_size = step_size
        self.discount = discount
        self.random = generator
        self.sum_reward = 0
        self.steps = 0
        self.last_state = None
        self.last_action = None
        self.device=device
    
    def tensorboard_close(self):
        self.writer.close()
    
    def policy(self, state: torch.Tensor):
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps / 1000)
        # print(eps)
        action = 0
        if self.random.random() < eps:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                action_values = self.predict_net(state).squeeze()
            self.writer.add_scalars("activation", {
                'left': action_values[0].item(),
                'right': action_values[1].item(),
                # 'down': action_values[2].item(),
                # 'no': action_values[3].item()
            }, self.steps)
            action = argmax(action_values, self.random)
        self.last_action = action
        return action
    
    
    def optimize_model(self):
        if not self.experiences.canSample():
            return
        batch_exprience = self.experiences.sample()
                    
        batch_last_state, batch_last_action, batch_reward, batch_new_state, batch_terminated = list(zip(*batch_exprience))
        
        batch_last_action = torch.tensor(batch_last_action, dtype=int)
        batch_last_action = batch_last_action[None, :]
        batch_last_state = torch.tensor(batch_last_state)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
        batch_new_state = torch.tensor(batch_new_state)
        batch_terminated = torch.tensor(batch_terminated, dtype=torch.float32)

        self.predict_net.train()
        
        q_current = self.predict_net(batch_last_state).gather(1, batch_last_action).squeeze()  
        
        with torch.no_grad():
            q_next = self.target_net(batch_new_state).max(1).values
        
        q_target = (batch_reward + (self.discount * q_next * batch_terminated))
        
        criterion = torch.nn.MSELoss()
        loss = criterion(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # self.writer.add_scalar('loss/e', loss, e)
    
    
    def train(self, episodes, n_iter, render_every):
        for e in (range(episodes)):
            terminated = False
            truncated = False
            state, _ = self.env.reset()
            self.sum_reward = 0
            self.last_state = state
            
            if e % n_iter == 0:
                self.target_net.load_state_dict(self.predict_net.state_dict())
                
            while not (terminated or truncated): 
                action = self.policy(torch.tensor(self.last_state))
                new_state, reward, terminated, truncated, info = self.env.step(action)
                self.sum_reward += reward
                
                experience = (self.last_state, action, np.float32(reward), new_state, not (terminated or truncated))
                self.experiences.add(experience)
                
                self.optimize_model()
                    
                # self.last_action = action
                self.last_state = new_state
            
            self.writer.add_scalar('total reward/episode', self.sum_reward, e)
                
        
            