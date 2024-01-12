"""
Author: Minh Pham-Dinh
Created: Dec 25th, 2023
Last Modified: Jan 10th, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Proximal Policy Optimization - vectorized environment.
    Inspired by Costa-Huang wandb implementation
    
    The implementation is based on:
    J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," 
    arXiv preprint arXiv:1707.06347, 2017. [Online]. Available: https://arxiv.org/abs/1707.06347
"""

import torch
import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
from Network import CNNnetwork
from tqdm import tqdm
from utils import PPOMemoryMultis
from torch.utils.data import DataLoader
import torch.nn as nn
from datetime import datetime
import itertools

from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv
)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent():
    def __init__(
            self,
            writer,
            device,
            envs: gym.Env,
            generator: np.random.Generator,
            num_envs: int = 1,
            update_epochs: int = 10,
            lr_actor: float = 0.01,
            lr_critic: float = 0.01,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            gae_lambda: float = 0.95,
            discount: float = 0.99,
            memory_size: int = 1028,
            mini_batch_size: int = 64,
            epsilon: float = 0.2) -> None:
        """ Main agent class for environment learning

        Args:
            writer (SummaryWriter): tensorboard summary writer for debugging
            device (torch.device): device to run agent on cpu, mps, or cuda
            env (gym.Env): gymnasium environment
            generator (np.random.Generator): seeded numpy random generator for reproducibly
            actor_net (torch.nn.Module): actor network
            critic_net (torch.nn.Module): critic network
            lr_actor (float, optional): actor learning rate. Defaults to 0.01.
            lr_critic (float, optional): critic learning rate. Defaults to 0.01.
            vf_coef (float, optional): value loss coefficient. Defaults to 0.5.
            ent_coef (float, optional): entropy loss coefficient. Defaults to 0.01.
            gae_lambda (float, optional): lambda coefficient for GAE. Defaults to 0.95.
            discount (float, optional): discount rate. Defaults to 0.99.
            memory_size (int, optional): size of memory (or batch size). Defaults to 1028.
            mini_batch_size (int, optional): size of mini batch. Defaults to 64.
            epsilon (float, optional): clipping range for PPO actor loss. Defaults to 0.2.
            num_envs (int): number of environment to run
        """
        self.envs = envs
        self.writer = writer
        self.memory = PPOMemoryMultis(memory_size, envs.single_observation_space.shape, num_envs, device=device)
        
        self.update_epochs = update_epochs
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        ).to(device)
        self.actor_net = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01).to(device)
        self.critic_net = layer_init(nn.Linear(512, 1), std=1).to(device)
        
        self.lr_actor = lr_actor
        self.optimizer_actor = torch.optim.Adam(itertools.chain(self.network.parameters(), self.actor_net.parameters()), lr_actor)
        
        self.lr_critic = lr_critic
        self.optimizer_critic = torch.optim.Adam(itertools.chain(self.network.parameters(), self.critic_net.parameters()), lr_critic)
        
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.mini_batch_size = mini_batch_size
        self.gae_lambda = gae_lambda
        self.discount = discount
        self.random = generator
        self.epsilon = epsilon
        self.steps = 0
        self.device = device
        self.next_states = None
        self.next_dones = torch.tensor([False] * num_envs).to(device, dtype=torch.float)

    def policy(self, states: torch.Tensor):
        """getting action based on current state

        Args:
            states (torch.Tensor): current states

        Returns:
            log_prob (Categorical): log_prob of current action
            action (int): action
        """
        with torch.no_grad():
            logits = self.actor_net(self.network(states/255.0))
        dists = torch.distributions.Categorical(logits=logits)
        actions = dists.sample()
        return dists.log_prob(actions), actions
    
    def get_value(self, states: torch.Tensor):
        """get critic value

        Args:
            states (torch.Tensor): current states

        Returns:
            critic_value
        """
        return self.critic_net(self.network(states/255.0))
    
    
    def compute_returns(self, values, dones, rewards):
        """Calculate returns with Generalized Advantage Estimation

        Args:
            values (torch.tensor): tensor of state values for bootstrapping
            dones (torch.tensor): boolean tensor of termination
            rewards (torch.tensor): rewards tensor

        Returns:
            returns
        """
        with torch.no_grad():
            next_value = self.get_value(self.next_states).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(advantages.shape[0])):
                if t == advantages.shape[0] - 1:
                    nextnonterminal = 1.0 - self.next_dones
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.discount * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.discount * self.gae_lambda * nextnonterminal * lastgaelam
        return advantages
 
    def optimize_models(self, epochs):
        """optimize models

        Args:
            epochs (int): number of epochs to perform gradients optimize
        """
        memory_loader = DataLoader(self.memory, batch_size=self.mini_batch_size, shuffle=True)
        
        states, actions, rewards, probs, dones, values = self.memory.get_data()
        
        advantages = self.compute_returns(values, dones, rewards)
        returns = advantages + values
        
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_states = states.reshape((-1, ) + self.envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_probs = probs.reshape(-1)
        
        for _ in (range(epochs)):
            frac = 1.0 - (_ - 1.0) / epochs
            lrnow = frac * self.lr_actor
            self.actor_net.param_groups[0]["lr"] = lrnow
            self.critic_net.param_groups[0]["lr"] = lrnow
            
            for batch in (memory_loader):
                mb_states = b_states[batch]
                mb_actions = b_actions[batch]
                mb_advantages = b_advantages[batch]
                mb_returns = b_returns[batch]
                
                new_values = self.get_value(mb_states).squeeze()
                logits = self.actor_net(self.network(mb_states/255.0))
                
                action_dists = torch.distributions.Categorical(logits=logits)
                
                new_log_probs = action_dists.log_prob(mb_actions)
                old_log_probs = b_probs[batch]
                
                entropy_loss = action_dists.entropy().mean()
                
                logratio = (new_log_probs - old_log_probs)
                ratio = logratio.exp()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                
                clipped_weighted_probs = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages
                actor_loss = -torch.min(ratio * mb_advantages, clipped_weighted_probs).mean()
                
                critic_loss = self.vf_coef * ((mb_returns - new_values)**2).mean()

                total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
                
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
        """ Training for a certain number of episodes

        Args:
            max_steps (int): number of steps to train on
        """
        states, _ = self.envs.reset()
        self.next_states = torch.tensor(states).to(device, dtype=torch.float)
        next_states_tensor = self.next_states
        pbar = tqdm(total=max_steps)
        while self.steps < max_steps:
            t = 0
            while True:
                t += 1
                pbar.update(num_envs)
                self.steps += num_envs
                log_probs, actions = self.policy(next_states_tensor)
                with torch.no_grad():
                    value = self.get_value(next_states_tensor).squeeze()
                
                next_states, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
                
                next_states_tensor = torch.tensor(next_states).to(device, dtype=torch.float)
                rewards_tensor = torch.tensor(rewards).to(device, dtype=torch.float)
                dones_tensor = torch.tensor(terminations | truncations).to(device, dtype=torch.float)
                
                self.memory.add(self.next_states, actions, log_probs, rewards_tensor, self.next_dones, value)
                
                self.next_states = next_states_tensor
                self.next_dones = dones_tensor
                
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            episodic_return = info['episode']['r']
                            writer.add_scalar('Performance/total reward over episodes', episodic_return, self.steps)
                    
                if self.memory.canOptimize():
                    break
            
            self.optimize_models(self.update_epochs)


    def eval(self, gym_id, episodes=10, verbose=False):
        """For evaluation, perform agent rollout under given policy with no optimization

        Args:
            episodes (int, optional): number of episodes to roll out. Defaults to 10.
            verbose (bool, optional): whether to print reward at each episode. Defaults to False.

        Returns:
            average_reward: average reward over episodes
        """
        env = gym.make(gym_id)
        mean_reward = 0
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                _, action = self.policy(torch.tensor(state))
                next_state, reward, terminated, truncated, _ = env.step(
                    action.item())
                done = (terminated or truncated)
                total_reward += reward
                state = next_state

            mean_reward += total_reward
            if verbose:
                print(
                    f"Episode {episode}, Total Reward: {total_reward}")
        return mean_reward / episodes

def make_env(gym_id, seed, idx, capture_video, video_record_freq, logpath):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx==0:
            env = gym.wrappers.RecordVideo(env, logpath + "/videos", episode_trigger= lambda t : t % video_record_freq == 0)
        
        #This wrapper samples initial states by taking a random number (between 1 and 30) of no-ops on reset.
        #for injecting stochasticity into the environment (Mnih et al., 2015) 
        env = NoopResetEnv(env)
        
        #This wrapper skips 4 frames by default, repeats the agent’s last action on the skipped frames, and sums up the rewards in 
        #the skipped frames. Such frame-skipping technique could considerably speed up the algorithm because the environment step is computationally cheaper than the agent’s forward pass (Mnih et al., 2015).
        env = MaxAndSkipEnv(env)
        
        #In the games where there are a life counter such as breakout, this wrapper marks the end of life as the end of episode.
        env = EpisodicLifeEnv(env)
        
        #This wrapper takes the FIRE action on reset for environments that are fixed until firing.
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        #This wrapper bins reward to {+1, 0, -1} by its sign
        env = ClipRewardEnv(env)
        
        #Change to grayscale and resize
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStack(env, 4)
        
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


if __name__ == "__main__":
    exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    gym_id = 'ALE/Breakout-v5'
    lr_actor = 2.5e-4
    lr_critic = 2.5e-4
    seed = 1
    max_steps = 10000000
    num_envs = 8    
    memory_size = 128
    minibatch_size = 265
    # device = torch.device('mps')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    capture_video=True
    video_record_freq = 200
    update_epochs = 4  
    eval_episodes = 50
    
    clip_coef = 0.1
    discount = 0.99
    gae_lambda = 0.95
    ent_coef = 0.01
    vf_coef = 0.5
    
    #wandb
    wandb_track = True
    wandb_project_name = 'Atari-PPO'
    wandb_entity = 'phdminh01'
    
    if wandb_track:
        import wandb
        
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            name=exp_name,
            monitor_gym=True,
            save_code=True,
        )
    
    logpath = f'./runs/PPO_atari/{gym_id}/{exp_name}' 
    
    envs = gym.vector.SyncVectorEnv([make_env(gym_id, seed, i, capture_video, video_record_freq, logpath) for i in range(num_envs)])
    
    #seeding
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator, seed = gym.utils.seeding.np_random(seed)
    
    # setup tensorboard
    writer = SummaryWriter(logpath)
    
    hparams = {
        'algorithm': 'PPO',
        'environment_id': gym_id,
        'num_envs': num_envs,
        'learning_rate_actor': lr_actor,
        'learning_rate_critic': lr_critic,
        'random_seed': seed,
        'maximum_steps': max_steps,
        'memory_size': memory_size,
        'minibatch_size': minibatch_size,
        'update_epochs': update_epochs,
        'clip_coefficient': clip_coef,
        'discount_factor': discount,
        'gae_lambda': gae_lambda,
        'entropy_coefficient': ent_coef,
        'value_function_coefficient': vf_coef,
        'eval_episodes': eval_episodes
    }

    
    # Initialize the Agent with specific hyperparameters
    agent = Agent(
        envs=envs,
        num_envs=num_envs,
        writer=writer,
        discount=discount,
        gae_lambda=gae_lambda,
        update_epochs=update_epochs,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        memory_size=memory_size,
        mini_batch_size=minibatch_size,
        epsilon=clip_coef,
        generator=generator,
        device=device
    )

    # Train the agent
    agent.train(max_steps)


    # Evaluation and save metrics
    metrics = {
        'reward_eval': agent.eval(episodes=eval_episodes, gym_id=gym_id)
    }

    writer.add_hparams(hparams, metrics)
    
    # Close the environment if necessary
    envs.close()