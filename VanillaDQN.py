"""
Author: Minh Pham-Dinh
Created: Dec 25th, 2023
Last Modified: Jan 1st, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Vanilla Deep Q Network.
    
    The implementation is based on:
    V. Mnih et al., "Human-level control through deep reinforcement learning," in Nature, 
    vol. 518, no. 7540, pp. 529-533, Feb. 2015. doi: 10.1038/nature14236. 
    Available: https://www.nature.com/articles/nature14236
"""

import torch
import copy
from utils import ReplayBuffer, argmax, parse_hyperparameters
import gymnasium as gym
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from Network import Net
import itertools
from tqdm import tqdm


class Agent():
    def __init__(
            self,
            writer,
            device,
            env: gym.Env,
            generator: np.random.Generator,
            net: torch.nn.Module,
            eps_start: float = 0.9,
            eps_decay: float = 1000,
            eps_end: float = 0.05,
            lr: float = 0.01,
            discount: float = 0.99,
            buffer_size: int = 10000,
            batch_size: int = 10000) -> None:
        self.env = env
        self.target_net = copy.deepcopy(net).to(device)
        self.predict_net = net.to(device)
        self.optimizer = torch.optim.AdamW(
            self.predict_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.experiences = ReplayBuffer(
            random=generator,
            buffer_size=buffer_size,
            batch_size=batch_size)
        self.writer = writer

        self.lr = lr
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.eps = 0

        self.discount = discount
        self.random = generator
        self.steps = 0
        self.device = device

    def policy(self, state: torch.Tensor):
        self.steps += 1
        self.eps = self.eps_end + \
            (self.eps_start - self.eps_end) * np.exp(-1. * self.steps / self.eps_decay)
        if self.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                action_values = self.predict_net(state).squeeze()
            return argmax(action_values, self.random)

    def optimize_model(self):
        if len(self.experiences) < self.experiences.batch_size:
            return

        states, actions, rewards, next_states, dones = self.experiences.sample()

        states = states.to(self.device, dtype=torch.float)
        next_states = next_states.to(self.device, dtype=torch.float)
        actions = actions.to(self.device, dtype=torch.long)
        rewards = rewards.to(self.device, dtype=torch.float)
        dones = dones.to(self.device)
        self.predict_net.train()

        # Current Q values
        q_current = self.predict_net(states).gather(1, actions)

        # Next Q values based on target network
        with torch.no_grad():
            q_next = self.target_net(next_states).detach().max(
                1).values.unsqueeze(-1)
            q_next[dones] = 0.0
        q_target = rewards + (self.discount * q_next)

        # Compute loss and update model
        criterion = torch.nn.MSELoss()
        loss = criterion(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, tau):
        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.policy(torch.tensor(state).to(self.device))
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = (terminated or truncated)
                total_reward += reward
                state_tensor = state
                next_state_tensor = next_state
                action_tensor = [action]
                reward_tensor = [reward]
                terminated_tensor = [done]

                self.experiences.add(
                    state_tensor,
                    action_tensor,
                    reward_tensor,
                    next_state_tensor,
                    terminated_tensor)

                state = next_state
                self.optimize_model()
            if episode % tau == 0:
                self.target_net.load_state_dict(self.predict_net.state_dict())

            self.writer.add_scalar(
                'total reward over episodes', total_reward, episode)
        torch.save(
            self.predict_net.state_dict(),
            f'./runs/VanillaDQN/tau_{tau}_batch_{self.batch_size}_epsdecay_{self.eps_decay}_lr_{self.lr}/saved_weights')

    def eval(self, episodes=10, verbose=False):
        mean_reward = 0
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    action_values = self.predict_net(torch.tensor(state).to(self.device)).squeeze()
                action = argmax(action_values, self.random)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = (terminated or truncated)
                total_reward += reward
                state = next_state

            mean_reward += total_reward
            if verbose:
                print(
                    f"Episode {episode}, Total Reward: {total_reward}, eps:  {self.eps}")
        return mean_reward / episodes


if __name__ == "__main__":
    device = torch.device('mps')
    env = gym.make("CartPole-v1", render_mode="human")
    action_space = env.action_space
    env.reset()
    obs_space = env.observation_space
    generator, seed = gym.utils.seeding.np_random(0)

    tau = 100
    batch_size = 64
    learning_rate = 1e-4
    epsilon_decay = 5000
    
    writer = SummaryWriter(
        log_dir=f'runs/VanillaDQN/tau_{tau}_batch_{batch_size}_epsdecay_{epsilon_decay}_lr_{learning_rate}')
    # Initialize the Agent with specific hyperparameters
    agent = Agent(
        env=env,
        net=Net(np.prod(*obs_space.shape), action_space.n, [128, 128]),
        writer=writer,
        discount=0.99,
        eps_start=1,
        eps_decay=epsilon_decay,
        eps_end=0.01,
        batch_size=batch_size,
        lr=learning_rate,
        generator=generator,
        device=device
    )

    # Train the agent
    agent.train(1000, tau)

    # Close the environment if necessary
    env.close()
