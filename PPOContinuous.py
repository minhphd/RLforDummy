"""
Author: Minh Pham-Dinh
Created: Dec 25th, 2023
Last Modified: Jan 10th, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Proximal Policy Optimization - vectorized environment.
    
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
from utils import PPOMemoryMultis
from torch.utils.data import DataLoader
from datetime import datetime
import itertools

class ActorNet(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, hiddens):
        super(ActorNet, self).__init__()
        self.actor_mean = Net(np.prod(*obs_shape), np.prod(*act_shape), hiddens).to(device)
        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape))).to(device)
        
    def forward(self, x):
        mean = self.actor_mean(x)
        log_std = self.actor_logstd.expand_as(mean)
        action_std = torch.exp(log_std)
        probs = torch.distributions.Normal(mean, action_std)
        return probs

class Agent():
    def __init__(
            self,
            writer,
            device,
            envs: gym.Env,
            generator: np.random.Generator,
            actor_net: torch.nn.Module,
            critic_net: torch.nn.Module,
            num_envs: int = 1,
            update_epochs: int = 10,
            lr: float = 2.5e-4,
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
        self.memory = PPOMemoryMultis(memory_size, envs.single_observation_space.shape, num_envs, device=device, continuous=True, act_shape=envs.single_action_space.shape)
        
        self.update_epochs = update_epochs
        
        self.actor_net = actor_net.to(device)
        self.critic_net = critic_net.to(device)
        
        self.lr = lr
        self.optimizer = torch.optim.AdamW(itertools.chain(self.actor_net.parameters(), self.critic_net.parameters()))
        
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

    def policy(self, states: torch.Tensor, action=None):
        """getting action based on current state

        Args:
            states (torch.Tensor): current states

        Returns:
            log_prob (Categorical): log_prob of current action
            action (int): action
        """
        probs = self.actor_net(states)
        if action is None:
            action = probs.sample()
        return probs.log_prob(action).sum(1), action, probs.entropy()
    
    
    def compute_returns(self, values, dones, rewards):
        """Calculate returns with Generalized Advantage Estimation

        Args:
            values (torch.tensor): tensor of state values for bootstrapping
            dones (torch.tensor): boolean tensor of termination
            rewards (torch.tensor): rewards tensor

        Returns:
            returns
        """
        n = values.shape[0]
        deltas = rewards[:-1] + self.discount * values[1:] * (1 - dones[:-1]) - values[:-1]

        advantages = torch.zeros_like(rewards, device=self.device, dtype=torch.float)        
        gae = torch.tensor(0.0, device=self.device, dtype=torch.float)
        for t in reversed(range(n - 1)):
            gae = deltas[t] + self.discount * self.gae_lambda * gae
            advantages[t] = gae
                
        with torch.no_grad():
            last_value = self.critic_net(self.next_states).squeeze()
        advantages[-1] = rewards[-1] + self.discount * last_value * (1 - self.next_dones[-1]) - values[-1] + self.discount * self.gae_lambda * gae
            
        return advantages
        
 
    def optimize_models(self, epochs):
        """optimize models

        Args:
            epochs (int): number of epochs to perform gradients optimize
        """
        memory_loader = DataLoader(self.memory, batch_size=self.mini_batch_size, shuffle=True)
        
        states, actions, rewards, probs, dones, values = self.memory.get_data()
        
        advantages = self.compute_returns(values, dones, rewards)
        returns = advantages + values.detach()
        
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_states = states.reshape((-1, ) + self.envs.single_observation_space.shape)
        b_actions = actions.reshape((-1, ) + self.envs.single_action_space.shape)
        b_rewards = rewards.reshape(-1)
        b_probs = probs.reshape(-1)
        b_values = values.reshape(-1)
        
        for _ in (range(epochs)):
            for batch in (memory_loader):
                mb_states = b_states[batch]
                mb_actions = b_actions[batch]
                mb_advantages = b_advantages[batch]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_returns = b_returns[batch]
                
                new_values = self.critic_net(mb_states).squeeze()
                new_log_probs, _, entropy = self.policy(mb_states, mb_actions)
                old_log_probs = b_probs[batch]
                
                entropy_loss = entropy.mean()
                
                logratio = (new_log_probs - old_log_probs)
                ratio = logratio.exp()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                
                clipped_weighted_probs = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages
                actor_loss = -torch.min(ratio * mb_advantages, clipped_weighted_probs).mean()
                
                critic_loss = 0.5 * ((mb_returns - new_values)**2).mean()
                
                total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
                
                # update net works 
                self.optimizer.zero_grad()
                total_loss.backward()               
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                self.optimizer.step()

            #log to tensorboard
        self.writer.add_scalar('Networks/critic_val', self.critic_net(states).mean(), self.steps)
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
                with torch.no_grad():
                    log_probs, actions, _ = self.policy(next_states_tensor)
                    value = self.critic_net(next_states_tensor).squeeze()
                
                next_states, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
                
                next_states_tensor = torch.tensor(next_states).to(device, dtype=torch.float)
                rewards_tensor = torch.tensor(rewards).to(device, dtype=torch.float)
                dones_tensor = torch.tensor(terminations | truncations).to(device, dtype=torch.float)
                
                self.memory.add(self.next_states, actions, log_probs, rewards_tensor, self.next_dones, value)
                
                self.next_states = next_states_tensor
                self.next_dones = dones_tensor
                
                if 'final_info' in infos.keys():
                    final_inf = infos['final_info']
                    avg_total_reward = np.zeros(len(final_inf))
                    for i in range(len(final_inf)):
                        if final_inf[i] is not None:
                            avg_total_reward[i] = final_inf[i]['episode']['r'][0]
                            
                    episodic_return = (np.mean(avg_total_reward[avg_total_reward != 0]))
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
        env = gym.make(gym_id, render_mode="human")
        mean_reward = 0
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                _, actions, _ = self.policy(torch.tensor([state], dtype=torch.float))
                next_state, reward, terminated, truncated, _ = env.step(actions.cpu().squeeze().numpy())
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
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    gym_id = 'HalfCheetah-v4'
    lr = 3e-4
    seed = 1
    max_steps = 2000000
    num_envs = 8    
    memory_size = 2048
    minibatch_size = 512
    device = torch.device('mps')
    capture_video=True
    video_record_freq = 50
    update_epochs = 10  
    eval_episodes = 50

    discount = 0.99
    gae_lambda = 0.95
    epsilon = 0.2
    ent_coef = 0.0
    vf_coef = 0.5
    
    #wandb
    wandb_track = False
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
        'learning_rate': lr,
        'random_seed': seed,
        'maximum_steps': max_steps,
        'memory_size': memory_size,
        'minibatch_size': minibatch_size,
        'update_epochs': update_epochs,
        'discount_factor': discount,
        'gae_lambda': gae_lambda,
        'epsilon': epsilon,
        'entropy_coefficient': ent_coef,
        'value_function_coefficient': vf_coef,
        'eval_episodes': eval_episodes
    }

    
    # Initialize the Agent with specific hyperparameters
    agent = Agent(
        envs=envs,
        num_envs=num_envs,
        actor_net=ActorNet(obs_shape=envs.single_observation_space.shape, act_shape=envs.single_action_space.shape, hiddens=[128, 128]),
        critic_net=Net(np.prod(*envs.single_observation_space.shape), 1, [128, 128]),
        writer=writer,
        discount=discount,
        gae_lambda=gae_lambda,
        update_epochs=update_epochs,
        lr=lr,
        memory_size=memory_size,
        mini_batch_size=minibatch_size,
        epsilon=epsilon,
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
