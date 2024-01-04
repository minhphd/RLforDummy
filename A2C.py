"""
Author: Minh Pham-Dinh
Created: Jan 2nd, 2024
Last Modified: Jan 1st, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Advantage Actor Critic.
    
    The implementation is based on:
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
            actor_net: torch.nn.Module,
            critic_net: torch.nn.Module,
            lr_actor: float = 0.01,
            lr_critic: float = 0.01,
            discount: float = 0.99) -> None:
        self.env = env
        self.writer = writer
        
        self.actor_net = actor_net.to(device)
        self.critic_net = critic_net.to(device)
        
        self.lr_actor = lr_actor
        self.optimizer_actor = torch.optim.AdamW(self.actor_net.parameters(), lr_actor)
        
        self.lr_critic = lr_critic
        self.optimizer_critic = torch.optim.AdamW(self.critic_net.parameters(), lr_critic)
        
        self.discount = discount
        self.random = generator
        self.steps = 0
        self.device = device

    def policy(self, state: torch.Tensor):
        with torch.no_grad():
            probs = self.actor_net(state)
        return torch.distributions.Categorical(probs).sample().item()
    
    def compute_returns(self, last_value, rewards):
        n = rewards.shape[0]
        returns = torch.zeros_like(rewards, device=self.device, dtype=torch.float)
        returns[-1] = rewards[-1] + self.discount * last_value

        for i in reversed(range(n-1)):
            returns[i] = rewards[i] + self.discount * returns[i + 1]
        
        return returns
                
    def optimize_models(self, states, actions, rewards, next_state, done):
        current_values = self.critic_net(states)
        if done:
            last_value = 0
        else:
            last_value = self.critic_net(next_state).detach().item()
            
        Rs = self.compute_returns(last_value, rewards)
        
        # print(Rs)
        
        action_dist = self.actor_net(states)
        log_probs = torch.log(action_dist)
        log_act = log_probs.gather(1, actions)
        
        advantages = Rs - current_values # or td_error
        entropy =  -(action_dist.detach() * log_probs.detach()).mean()
        
        actor_loss = -(log_act * advantages.detach()).mean() - 0.0005 * entropy
        critic_loss = (advantages**2).mean()
        
        self.writer.add_scalar('Networks/actor_loss', actor_loss, self.steps)
        self.writer.add_scalar('Networks/critic_loss', critic_loss, self.steps)
        self.writer.add_scalar('Networks/entropy_loss', entropy, self.steps)
        
        
        # Update networks
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor_net.parameters(), 100)
        self.optimizer_actor.step()
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic_net.parameters(), 100)
        self.optimizer_critic.step()

        
    def train(self, episodes, n):
        #n = number of step go before start agrevating data
        states = []
        rewards = []
        actions = []
        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            state_tensor = torch.tensor(state).to(self.device, dtype=torch.float)
            t = 0
            while not done:
                self.steps += 1
                t += 1
                
                #get action from actor net
                action = self.policy(state_tensor)
                
                #act
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                next_state_tensor = torch.tensor(next_state).to(self.device, dtype=torch.float)
                reward_tensor = torch.tensor([reward]).to(self.device, dtype=torch.float)
                
                states.append(state_tensor)
                rewards.append(reward_tensor)
                actions.append(torch.tensor([action]).to(self.device, dtype=torch.long))
                
                if done or t % n == 0:
                    self.optimize_models(torch.stack(states), torch.stack(actions), torch.stack(rewards), next_state_tensor, done)  
                    states = []
                    rewards = []      
                    actions = []
                
                state_tensor = next_state_tensor
                total_reward += reward
                
            # print(
            #     f"Episode {episode}, Total Reward: {total_reward}")
            self.writer.add_scalar(
                'Performance/total reward over episodes', total_reward, episode)
        
            for name, param in self.actor_net.named_parameters():
                self.writer.add_histogram(f'ActorNet/{name}', param, self.steps)
                if param.requires_grad:
                    self.writer.add_histogram(f'ActorNet/{name}_grad', param.grad, self.steps)

            for name, param in self.critic_net.named_parameters():
                self.writer.add_histogram(f'CriticNet/{name}', param, self.steps)
                if param.requires_grad:
                    self.writer.add_histogram(f'CriticNet/{name}_grad', param.grad, self.steps)
            
            if episode % 100 == 0:
                torch.save(self.actor_net, f'./runs/A3C/saved_model{episode}')
        
    def eval(self, episodes=10, verbose=False):
        mean_reward = 0
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                with torch.no_grad():
                    action_values = self.actor_net(torch.tensor(state).to(self.device)).squeeze()
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
    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v1", render_mode="human")
    action_space = env.action_space
    env.reset()
    obs_space = env.observation_space
    generator, seed = gym.utils.seeding.np_random(0)
    
    device = torch.device("mps")
    
    writer = SummaryWriter()
    # Initialize the Agent with specific hyperparameters
    agent = Agent(
        env=env,
        actor_net=Net(np.prod(*obs_space.shape), action_space.n, [128, 128], softmax=True),
        critic_net=Net(np.prod(*obs_space.shape), 1, [128, 128]),
        writer=SummaryWriter(),
        discount=0.99,
        lr_actor=0.001,
        lr_critic=0.001,
        generator=generator,
        device=device
    )

    # Train the agent
    agent.train(10, 16)

    # Close the environment if necessary
    env.close()
