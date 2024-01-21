"""
Author: Minh Pham-Dinh
Created: Jan 2nd, 2024
Last Modified: Jan 4th, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Advantage Actor Critic.
    
    The implementation is based on:
    V. Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning," 2016. 
    [Online]. Available: https://arxiv.org/abs/1602.01783v2
"""

import torch
import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
from misc.Network import Net
from tqdm import tqdm
from datetime import datetime


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
            discount: float = 0.99,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01) -> None:
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
        """
        
        self.env = env
        self.writer = writer
        
        self.actor_net = actor_net.to(device)
        self.critic_net = critic_net.to(device)
        
        self.lr_actor = lr_actor
        self.optimizer_actor = torch.optim.AdamW(self.actor_net.parameters(), lr_actor)
        
        self.lr_critic = lr_critic
        self.optimizer_critic = torch.optim.AdamW(self.critic_net.parameters(), lr_critic)
        
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.discount = discount
        self.random = generator
        self.steps = 0
        self.device = device


    def policy(self, state: torch.Tensor):
        """getting action based on current state

        Args:
            state (torch.Tensor): current state

        Returns:
            action (int): action
        """
        with torch.no_grad():
            probs = self.actor_net(state)
        return torch.distributions.Categorical(probs).sample().item()
    
    
    def compute_returns(self, last_value, rewards):
        """Calculate the returns by bootstrapping

        Args:
            last_value (torch.Tensor): last state value before start calculating
            rewards (torch.Tensor): list of reward
            
        Returns:
            action (int): action
        """
        n = rewards.shape[0]
        returns = torch.zeros_like(rewards, device=self.device, dtype=torch.float)
        returns[-1] = rewards[-1] + self.discount * last_value

        for i in reversed(range(n-1)):
            returns[i] = rewards[i] + self.discount * returns[i + 1]
        
        return returns
                
                
    def optimize_models(self, states, actions, rewards, next_state, done):
        """optimize models

        Args:
            epochs (int): number of epochs to perform gradients optimize
        """
        current_values = self.critic_net(states)
        if done:
            last_value = 0
        else:
            last_value = self.critic_net(next_state).detach().item()
        
        # compute returns    
        Rs = self.compute_returns(last_value, rewards)
        
        # compute log activation
        action_dist = self.actor_net(states)
        dist = torch.distributions.Categorical(action_dist)
        log_act = dist.log_prob(actions.squeeze()).unsqueeze(1)
        
        # compute losses
        advantages = Rs - current_values # or td_error
        entropy =  -(action_dist.detach() * torch.log(action_dist).detach()).mean()
        actor_loss = -(log_act * advantages.detach()).mean()
        critic_loss = (advantages**2).mean()
        total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
        
        self.writer.add_scalar('Networks/actor_loss', actor_loss, self.steps)
        self.writer.add_scalar('Networks/critic_loss', critic_loss, self.steps)
        self.writer.add_scalar('Networks/entropy_loss', entropy, self.steps)
        self.writer.add_scalar('Networks/total_loss', total_loss, self.steps)
        
        
        # update net works 
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad() 
        total_loss.backward()               
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
        self.optimizer_critic.step()
        self.optimizer_actor.step()


    def train(self, episodes, n):
        """Rollout and train the model for a certain number of episodes

        Args:
            episodes (int): number of training episodes
            n (int): number of rollout steps before start optimizing model
        """
        
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
                
            self.writer.add_scalar(
                'Performance/total reward over episodes', total_reward, episode)
            
        
    def eval(self, episodes=10, verbose=False):
        mean_reward = 0
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.policy(torch.tensor(state))
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
    # Experiment setup
    exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')  # Unique experiment name based on current timestamp
    gym_id = 'LunarLander-v2'  # Environment ID for Gym (LunarLander-v2)
    lr_actor = 2.5e-4  # Learning rate for the actor network
    lr_critic = 1e-3  # Learning rate for the critic network
    seed = 1  # Seed for reproducibility
    max_episodes = 1000  # Maximum number of episodes

    # Rollout and training parameters
    rollout_steps = 516  # Number of rollout steps
    device = torch.device('cpu')  # Device for training (CPU)
    capture_video = True  # Flag to determine whether to capture videos
    video_record_freq = 200  # Frequency of recording video episodes
    eval_episodes = 50  # Number of episodes for evaluation

    # A2C hyperparameters
    discount = 0.99  # Discount factor for future rewards
    ent_coef = 0.01  # Entropy coefficient
    vf_coef = 0.5  # Value function coefficient

    # Weights & Biases configuration (for experiment tracking)
    wandb_track = False  # Flag to enable/disable Weights & Biases tracking
    wandb_project_name = 'A2C'  # Project name in Weights & Biases
    wandb_entity = 'phdminh01'  # User/entity name in Weights & Biases
    
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
    
    logpath = f'./runs/A2C/{gym_id}/{exp_name}' 
    
    env = gym.make(gym_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, logpath + "/videos", episode_trigger= lambda t : t % video_record_freq == 0)
    
    #seeding
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator, seed = gym.utils.seeding.np_random(seed)
    
    
    # setup tensorboard
    writer = SummaryWriter(logpath)
    
    hparams = {
        'algorithm': 'A2C',
        'environment_id': gym_id,
        'learning_rate_actor': lr_actor,
        'learning_rate_critic': lr_critic,
        'random_seed': seed,
        'rollout_steps': rollout_steps,
        'maximum_episodes': max_episodes,
        'discount_factor': discount,
        'entropy_coefficient': ent_coef,
        'value_function_coefficient': vf_coef,
        'eval_episodes': eval_episodes
    }

    
    # Initialize the Agent with specific hyperparameters
    agent = Agent(
        env=env,
        actor_net=Net(np.prod(*env.observation_space.shape), env.action_space.n, [128, 128], softmax=True),
        critic_net=Net(np.prod(*env.observation_space.shape), 1, [128, 128]),
        writer=writer,
        discount=discount,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        generator=generator,
        device=device
    )

    # Train the agent
    agent.train(max_episodes, rollout_steps)

    # Evaluation and save metrics
    metrics = {
        'reward_eval': agent.eval(episodes=eval_episodes)
    }

    writer.add_hparams(hparams, metrics)
    
    # Close the environment if necessary
    env.close()