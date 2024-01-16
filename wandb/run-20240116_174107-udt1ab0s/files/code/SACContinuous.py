"""
Author: Minh Pham-Dinh
Created: Jan 14th, 2024
Last Modified: Jan 14th, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Soft Actor Critic - intend to use with continuous environment.
    
    The implementation is based on:
    Reference: T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft Actor-Critic: Off-Policy 
    Deep Reinforcement Learning with Real-World Robots," arXiv preprint arXiv:1801.01290, 2018. 
    (https://arxiv.org/abs/1707.06347)
"""

import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from Network import Net
from tqdm import tqdm
from utils import SACMemory
from torch.utils.data import DataLoader
from datetime import datetime
import itertools
import copy

LOGSTD_LOW = -5
LOGSTD_HIGH = 2

class ActorNet(torch.nn.Module):
    def __init__(self, env, obs_shape, act_shape, hiddens):
        super().__init__()
        self.fc = Net(np.prod(*obs_shape), hiddens[-1], hiddens, open_ended=True)
        
        self.fc_mean = torch.nn.Sequential(
            torch.nn.Linear(hiddens[-1], np.prod(act_shape)))
        
        self.fc_logstd = torch.nn.Sequential(
            torch.nn.Linear(hiddens[-1], np.prod(act_shape)))
        
        #from CleanRL SAC implementation
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float)
        )
        
    def forward(self, x):
        """_summary_

        Args:
            x (torch.tensor): state

        Returns:
            action: action given sampled from network
            log_prob: log probability of action
            mean: mean of action value
        """
        fc_out = self.fc(x)
        mean = self.fc_mean(fc_out)
        log_std = torch.tanh(self.fc_logstd(fc_out))
        
        # log_std clipping, https://liuliu.me/eyes/5-more-implementation-details-of-ppo-and-sac/
        log_std = LOGSTD_LOW + 0.5 * (LOGSTD_HIGH - LOGSTD_LOW) * (log_std + 1)
        action_std = log_std.exp()
        
        probs = torch.distributions.Normal(mean, action_std)
        
        #reparameterization trick
        x_t = probs.rsample()
        y_t = torch.tanh(x_t)
        
        action = y_t * self.action_scale + self.action_bias
        log_prob = probs.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class SoftQNetwork(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, hiddens):
        super().__init__()
        self.fc = Net(np.prod(*obs_shape) + np.prod(act_shape), 1, hiddens)

    def forward(self, x, a):
        """forward through the q network

        Args:
            x (torch.tensor): states
            a (torch.tensor): actions

        Returns:
            q_value: Q value
        """
        x = torch.cat([x, a], 1)
        return self.fc(x)


class Agent():
    def __init__(
            self,
            writer,
            device,
            env: gym.Env,
            generator: np.random.Generator,
            policy_net: torch.torch.nn.Module,
            q_net: torch.torch.nn.Module,
            update_epochs: int = 10,
            policy_lr: float = 3e-4,
            q_lr:float = 1e-3,
            gamma: float = 0.99,
            memory_size: int = 1028,
            mini_batch_size: int = 64,
            tau: float = 0.5) -> None:
        self.env = env
        self.steps = 0
        self.device = device
        
        self.writer =writer
        self.generator=generator
        self.memory = SACMemory(memory_size, env, generator, device)
        self.memory_size = memory_size
        self.mini_batch_size = mini_batch_size
        self.update_epochs = update_epochs
        
        self.policy_net = policy_net.to(device)
        self.q1_net = copy.deepcopy(q_net).to(device)
        self.q2_net = copy.deepcopy(q_net).to(device)
        self.q1_target_net = copy.deepcopy(q_net).to(device)
        self.q2_target_net = copy.deepcopy(q_net).to(device)
        
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = policy_lr)
                                          
        self.q_optimizer = torch.optim.Adam(itertools.chain(self.q1_net.parameters(),
                                                          self.q2_net.parameters()), lr=q_lr)
        
        self.gamma = gamma
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)
        self.tau = tau
 

    def optimize_models(self, epochs):
        for _ in range(epochs):
            mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = self.memory.sample(self.mini_batch_size)
            
            #update q
            with torch.no_grad():
                next_a, next_log, _ = self.policy_net(mb_next_states)
                q1_next = self.q1_target_net(mb_next_states, next_a).squeeze()
                q2_next = self.q2_target_net(mb_next_states, next_a).squeeze()
                target = (mb_rewards + self.gamma * (1-mb_dones) * (torch.min(q1_next, q2_next)  - self.alpha * next_log.squeeze()))
            
            q1_current = (self.q1_net(mb_states, mb_actions)).squeeze()
            q2_current = (self.q2_net(mb_states, mb_actions)).squeeze()
            
            q1_loss = F.mse_loss(q1_current, target)
            q2_loss = F.mse_loss(q2_current, target)
            
            q_loss = q1_loss + q2_loss
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            #update policy
            pi, logpi, _ = self.policy_net(mb_states)
            q1_pi = (self.q1_net(mb_states, pi)).squeeze()
            q2_pi = (self.q2_net(mb_states, pi)).squeeze()
            
            policy_loss = ((self.alpha * logpi).squeeze() - torch.min(q1_pi, q2_pi)).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            #update alpha
            with torch.no_grad():
                _, log_pi, _ = self.policy_net(mb_states)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            
            #update target net
            for param, target_param in zip(self.q1_net.parameters(), self.q1_target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2_net.parameters(), self.q2_target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
            #log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar('Networks/actor_loss', policy_loss, self.steps)
                self.writer.add_scalar('Networks/critic_loss', q_loss, self.steps)
                self.writer.add_scalar('Networks/entropy_loss', logpi.mean(), self.steps)
                self.writer.add_scalar('Networks/alpha_loss', alpha_loss, self.steps)
            # self.writer.add_scalar('Networks/KL divergence', approx_kl, self.steps)


    def train(self, max_steps, train_freq, lr_start):
        """ Training for a certain number of maximum steps
        
        Args:
            max_steps (int): number of steps to train on
            train_freq (int): start optimize net every 'train_freq' steps (global steps)
        """
        
        pbar = tqdm(total=max_steps)
        state, _ = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        for self.steps in range(max_steps):
            pbar.update()
            
            if self.steps < lr_start:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _, _ = self.policy_net(state_tensor.unsqueeze(0))
                    action = action.cpu().squeeze().numpy()
                
            next_states, rewards, termination, truncation, info = self.env.step(action)
            
            # start sampling action on cpu before learning to prevent bottle neck
            next_states_tensor = torch.tensor(next_states, dtype=torch.float).to(self.device)
            action_tensor = torch.tensor(action, dtype=torch.float).to(self.device)
            
            #add to memory
            self.memory.add(state_tensor, action_tensor, rewards, next_states_tensor, termination)
            state_tensor = next_states_tensor
            
            if 'episode' in info:
                if self.writer is not None:
                    writer.add_scalar('Performance/total reward over episodes', info['episode']['r'], self.steps)
                state, _ = self.env.reset()
                state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            
            if self.steps % train_freq == 0 and self.steps >= lr_start:     
                self.optimize_models(self.update_epochs)
            
                
        pbar.close()     

    def eval(self, episodes=10, path='./'):
        """For evaluation, perform agent rollout under given policy with no optimization

        Args:
            episodes (int, optional): number of episodes to roll out. Defaults to 10.
            verbose (bool, optional): whether to print reward at each episode. Defaults to False.

        Returns:
            average_reward: average reward over episodes
        """
        total_reward = 0
        for _ in range(episodes):
            state, info = self.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float).to(device)
            while 'episode' not in info:
                with torch.no_grad():
                    action, _, _ = self.policy_net(state_tensor.unsqueeze(0))
                    action = action.cpu().squeeze().numpy()
                next_state, _, _, _, info = self.env.step(action)
                state = next_state
            total_reward += info['episode']['r']
            state, _ = self.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        torch.save(self.policy_net, path)
        return total_reward/episodes
            

if __name__ == "__main__":
    # Experiment setup
    exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')  # Unique experiment name based on current timestamp
    gym_id = 'LunarLander-v2'  # Environment ID for Gym
    # Alternative environments:
    # gym_id = 'BipedalWalker-v3'
    # gym_id = 'Pendulum-v1'

    # Hyperparameters for SAC
    policy_lr = 3e-4  # Learning rate for the policy network
    q_lr = 1e-3  # Learning rate for the Q network
    gamma = 0.99  # Discount factor for future rewards
    tau = 0.005  # Target network update rate
    memory_size = int(1e6)  # Size of the replay buffer
    minibatch_size = 256  # Size of minibatches for training
    update_epochs = 10  # Number of epochs for updating networks
    max_steps = 500000  # Maximum number of steps to train
    train_freq = 100  # Frequency of training steps
    lr_start = 10000  # Step to start learning rate scheduling
    eval_episodes = 50  # Number of episodes for evaluation

    # Environment and training setup
    seed = 1  # Seed for reproducibility
    device = torch.device('cpu')  # Device for training (CPU, CUDA, or MPS)
    capture_video = True  # Flag to determine whether to capture videos
    video_record_freq = 200  # Frequency of recording video episodes

    # Weights & Biases configuration (for experiment tracking)
    wandb_track = True  # Flag to enable/disable Weights & Biases tracking
    wandb_project_name = 'SAC'  # Project name in Weights & Biases
    wandb_entity = 'phdminh01'  # User/entity name in Weights & Biases

    # Additional parameters (currently unused or commented out)
    verbose = True  # Verbosity flag, True if log to Tensorboard
    
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
    
    logpath = f'./runs/SAC/{gym_id}/{exp_name}' 
    
    env = gym.make(gym_id, render_mode="rgb_array", continuous=True)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, logpath + "/videos", episode_trigger= lambda t : t % video_record_freq == 0)
    env = gym.wrappers.ClipAction(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    #seeding
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator, seed = gym.utils.seeding.np_random(seed)
    
    # setup tensorboard
    if verbose:
        writer = SummaryWriter(logpath)
    else:
        writer = None
    
    hparams = {
        'algorithm': 'SAC',
        'environment_id': gym_id,
        'policy_lr': policy_lr,
        'q_lr': q_lr,
        'tau': tau,
        'random_seed': seed,
        'maximum_steps': max_steps,
        'memory_size': memory_size,
        'training_freq': train_freq,
        'minibatch_size': minibatch_size,
        'update_epochs (gradient steps)': update_epochs,
        'gamma': gamma,
        'eval_episodes': eval_episodes
    }

    
    # Initialize the Agent with specific hyperparameters
    agent = Agent(
        writer=writer,
        device=device,
        env=env,
        generator=generator,
        policy_net=ActorNet(env=env, obs_shape=env.observation_space.shape, act_shape=env.action_space.shape, hiddens=[256]),
        q_net=SoftQNetwork(obs_shape=env.observation_space.shape, act_shape=env.action_space.shape, hiddens=[256]),
        update_epochs=update_epochs,
        policy_lr=policy_lr,
        q_lr=q_lr,
        gamma=gamma,
        memory_size=memory_size,
        mini_batch_size=minibatch_size,
        tau=tau
    )

    # Train the agent
    agent.train(max_steps, train_freq, lr_start)

    # Evaluation and save metrics
    metrics = {
        'reward_eval': agent.eval(episodes=eval_episodes, path=(logpath + '/saved_model'))
    }

    writer.add_hparams(hparams, metrics)
    
    # Close the environment if necessary
    env.close()
