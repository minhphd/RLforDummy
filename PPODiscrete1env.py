"""
Author: Minh Pham-Dinh
Created: Jan 7th, 2024
Last Modified: Jan 10th, 2024
Email: mhpham26@colby.edu

Description:
    Network file for used with RL files
"""

import torch
import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
from Network import Net
from tqdm import tqdm
from utils import PPOMemory
from torch.utils.data import DataLoader
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
            logpath: str,
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
        """
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
        """getting action based on current state

        Args:
            state (torch.Tensor): current state

        Returns:
            log_prob (Categorical): log_prob of current action
            action (int): action
        """
        with torch.no_grad():
            probs = self.actor_net(state)
        dists = torch.distributions.Categorical(probs)
        action = dists.sample()
        return dists.log_prob(action), action.item()
    
    
    def compute_advantages(self, values, dones, rewards):
        """Calculate Generalized Advantage Estimation

        Args:
            values (torch.tensor): tensor of state values for bootstrapping
            dones (torch.tensor): boolean tensor of termination
            rewards (torch.tensor): rewards tensor

        Returns:
            advantages: Generalized Adavantage Estimation
        """
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
        """optimize models

        Args:
            epochs (int): number of epochs to perform gradients optimize
        """
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

                total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy_loss
                
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

        
        
    def train(self, episodes):
        """ Training for a certain number of episodes

        Args:
            max_steps (int): number of episodes to train on
        """
        for episode in tqdm(range(episodes)):
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
                
            self.writer.add_scalar(
                'Performance/total reward over episodes', total_reward, episode)
        
    def eval(self, episodes=10, verbose=False):
        """For evaluation, perform agent rollout under given policy with no optimization

        Args:
            episodes (int, optional): number of episodes to roll out. Defaults to 10.
            verbose (bool, optional): whether to print reward at each episode. Defaults to False.

        Returns:
            average_reward: average reward over episodes
        """
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
    exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    gym_id = 'LunarLander-v2'
    lr_actor = 2.5e-4
    lr_critic = 1e-3
    seed = 1
    max_episodes = 1500
    
    memory_size = 1028
    minibatch_size = 128
    device = torch.device('cpu')
    capture_video=True
    video_record_freq = 200
    update_epochs = 10  
    eval_episodes = 50

    clip_coef = 0.2
    discount = 0.99
    gae_lambda = 0.95
    epsilon = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    
    #wandb
    wandb_track = True
    wandb_project_name = 'PPO_1env_discrete'
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
    
    logpath = f'./runs/PPO/{gym_id}/{exp_name}' 
    
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
        'algorithm': 'PPO',
        'environment_id': gym_id,
        'learning_rate_actor': lr_actor,
        'learning_rate_critic': lr_critic,
        'random_seed': seed,
        'maximum_episodes': max_episodes,
        'memory_size': memory_size,
        'minibatch_size': minibatch_size,
        'update_epochs': update_epochs,
        'clip_coefficient': clip_coef,
        'discount_factor': discount,
        'gae_lambda': gae_lambda,
        'epsilon': epsilon,
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
        gae_lambda=gae_lambda,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        memory_size=memory_size,
        mini_batch_size=minibatch_size,
        epsilon=epsilon,
        generator=generator,
        device=device,
        logpath=logpath
    )

    # Train the agent
    agent.train(max_episodes)


    # Evaluation and save metrics
    metrics = {
        'reward_eval': agent.eval(episodes=eval_episodes)
    }

    writer.add_hparams(hparams, metrics)
    
    # Close the environment if necessary
    env.close()
