"""
Author: Minh Pham-Dinh
Created: Jan 7th, 2024
Last Modified: Jan 7th, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Proximal Policy Optimization - single environment.
    
    The implementation is based on:
    J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," 
    arXiv preprint arXiv:1707.06347, 2017. [Online]. Available: https://arxiv.org/abs/1707.06347
"""

import torch
import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
from Network import Net
from tqdm import tqdm
from utils import PPOMemory
from torch.utils.data import DataLoader

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
            gae_lambda: float = 0.95,
            discount: float = 0.99,
            memory_size: int = 1028,
            mini_batch_size: int = 64,
            epsilon: float = 0.2) -> None:
        self.env = env
        self.writer = writer
        self.memory = PPOMemory(memory_size, env.action_space.n, np.prod(*env.observation_space.shape), device=device)
        
        self.actor_net = actor_net.to(device)
        self.critic_net = critic_net.to(device)
        
        self.lr_actor = lr_actor
        self.optimizer_actor = torch.optim.Adam(self.actor_net.parameters(), lr_actor)
        
        self.lr_critic = lr_critic
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr_critic)
        
        self.mini_batch_size = mini_batch_size
        self.gae_lambda = gae_lambda
        self.discount = discount
        self.random = generator
        self.epsilon = epsilon
        self.steps = 0
        self.device = device
        self.next_state = None

    def policy(self, state: torch.Tensor):
        with torch.no_grad():
            probs = self.actor_net(state)
        dists = torch.distributions.Categorical(probs)
        action = dists.sample()
        return dists.log_prob(action), action.item()
    
    def compute_advantages(self, values, dones, rewards):
        n = values.shape[0]
        deltas = rewards[:-1] + self.discount * values[1:] * (1 - dones[:-1]) - values[:-1]

        advantages = torch.zeros_like(rewards, device=self.device, dtype=torch.float)        
        gae = torch.tensor(0.0, device=self.device, dtype=torch.float)
        for t in reversed(range(n - 1)):
            gae = deltas[t] + self.discount * self.gae_lambda * gae
            advantages[t] = gae
        
        with torch.no_grad():
            last_value = self.critic_net(self.next_state)
        advantages[-1] = rewards[-1] + self.discount * last_value * dones[-1] - values[-1] + self.discount * self.gae_lambda * gae

        return advantages
                
    def optimize_models(self, epochs):
        memory_loader = DataLoader(self.memory, batch_size=self.mini_batch_size, shuffle=True)
        
        _, states, _, rewards, dones = self.memory.get_data()
        
        with torch.no_grad():
            values = self.critic_net(states)
        
        advantages = self.compute_advantages(values, dones, rewards)
        for _ in (range(epochs)):
            for batch in (memory_loader):
                batch_states = batch['states']
                batch_actions = batch['actions'].squeeze()
                
                new_values = self.critic_net(batch_states)
                cur_probs = self.actor_net(batch_states)
                
                #normalize
                # batch_advantages = (advantages[batch['idx']] - advantages[batch['idx']].mean()) / (advantages[batch['idx']].std() + 1e-8)
                batch_advantages = advantages[batch['idx']]
                
                action_dists = torch.distributions.Categorical(cur_probs)
                
                new_log_probs = action_dists.log_prob(batch_actions).unsqueeze(1)
                old_log_probs = batch['probs']
                
                entropy_loss = action_dists.entropy().mean()
                
                logratio = (new_log_probs - old_log_probs)
                ratio = logratio.exp()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                
                clipped_weighted_probs = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(ratio * batch_advantages, clipped_weighted_probs).mean()
                
                returns = batch_advantages + values[batch['idx']].detach()
                critic_loss = ((returns - new_values)**2).mean()

                total_loss = actor_loss + 0.5*critic_loss - 0.005 * entropy_loss
                
                
                # update net works 
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad() 
                total_loss.backward()               
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                self.optimizer_critic.step()
                self.optimizer_actor.step()

            #log to tensorboard
            self.writer.add_scalar('Networks/actor_loss', actor_loss, self.steps)
            self.writer.add_scalar('Networks/critic_loss', critic_loss, self.steps)
            self.writer.add_scalar('Networks/entropy_loss', entropy_loss, self.steps)
            self.writer.add_scalar('Networks/total_loss', total_loss, self.steps)
            self.writer.add_scalar('Networks/KL divergence', approx_kl, self.steps)
            
        self.memory.clear()

        
        
    def train(self, max_steps):
        episode = 0
        while self.steps < max_steps:
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            state_tensor = torch.tensor(state).to(self.device, dtype=torch.float)
            episode += 1
            while not done:
                self.steps += 1
                
                #get action from actor net
                log_probs, action = self.policy(state_tensor)
                
                #act
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                next_state_tensor = torch.tensor(next_state).to(self.device, dtype=torch.float)
                reward_tensor = torch.tensor([reward]).to(self.device, dtype=torch.float)
                
                self.memory.add(state_tensor, action, log_probs, reward_tensor, done)
                
                if self.memory.canOptimize():
                    self.next_state = next_state_tensor
                    self.optimize_models(10)
                
                state_tensor = next_state_tensor
                total_reward += reward
                
            print(
                f"global step {self.steps}, Total Reward: {total_reward}")
            self.writer.add_scalar(
                'Performance/total reward over episodes', total_reward, episode)
                
            # if episode % 100 == 0:
            #     torch.save(self.actor_net, f'./runs/PPO/saved_model{episode}')
        
    def eval(self, episodes=10, verbose=False):
        mean_reward = 0
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                _, action = self.policy(torch.tensor(state))
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = (terminated or truncated)
                total_reward += reward
                state = next_state

            mean_reward += total_reward
            if verbose:
                print(
                    f"Episode {episode}, Total Reward: {total_reward}")
        return mean_reward / episodes


if __name__ == "__main__":
    # # env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v1")
    # action_space = env.action_space
    # env.reset()
    # obs_space = env.observation_space
    # generator, seed = gym.utils.seeding.np_random(0)
    
    # device = torch.device("cpu")
    
    # writer = SummaryWriter()
    # # Initialize the Agent with specific hyperparameters
    # agent = Agent(
    #     env=env,
    #     actor_net=Net(np.prod(*obs_space.shape), action_space.n, [128, 128], softmax=True),
    #     critic_net=Net(np.prod(*obs_space.shape), 1, [128, 128]),
    #     writer=SummaryWriter(),
    #     discount=0.99,
    #     gae_lambda=0.95,
    #     lr_actor=3e-3,
    #     lr_critic=1e-3,
    #     memory_size=516,
    #     mini_batch_size=128,
    #     epsilon=0.2,
    #     generator=generator,
    #     device=device
    # )

    # # Train the agent
    # agent.train(300000)

    # # Close the environment if necessary
    # env.close()



    #--------------------FOR EVALUATION------------------------

    # env = gym.make('CartPole-v1', render_mode="human")
    env = gym.make('LunarLander-v2', render_mode="human")
    action_space = env.action_space
    env.reset()
    obs_space = env.observation_space
    generator, seed = gym.utils.seeding.np_random(0)
    
    device = torch.device("cpu")
    
    trained_net = torch.load("runs/PPo/LunarLanderSol/LunarLanderSolution", map_location=device)
    # trained_net = torch.load("./runs/PPO/saved_model800", map_location=device)

    eval_agent = Agent(
        env=env,
        actor_net=trained_net,
        critic_net=trained_net,
        writer=None,
        device=device,
        generator=generator
    )

    print(eval_agent.eval(10, verbose=True))
    env.close()
