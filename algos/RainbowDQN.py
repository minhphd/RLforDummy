"""
Author: Minh Pham-Dinh
Created: Jan 1st, 2024
Last Modified: Jan 1st, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Rainbow Deep Q Network. 
    *This code only implemented prioritized replay and double DQN
    
    The implementation is based on:
    M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning," 
    arXiv preprint arXiv:1710.02298, Oct. 2017. 
    Available: https://arxiv.org/abs/1710.02298
"""

import torch
import copy
from misc.utils import PrioritizedReplayBuffer, argmax
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from misc.Network import Net
from tqdm import tqdm
from datetime import datetime
from misc.Network import Net


class Agent():
    def __init__(
            self,
            writer,
            device,
            env: gym.Env,
            generator: np.random.Generator,
            net: torch.nn.Module,
            beta: float = 0.4,
            beta_flourish: float = 0.00001,
            eps_start: float = 0.9,
            eps_decay: float = 1000,
            eps_end: float = 0.05,
            lr: float = 0.01,
            discount: float = 0.99,
            buffer_size: int = 10000,
            batch_size: int = 10000) -> None:
        """Create a RL agent in given environment

        Args:
            writer (SummaryWriter): Tensorboard writer to log data 
            device (torch.device): device type to perform on "mps", "cuda", "cpu"
            env (gym.Env): gymnasium environment
            generator (np.random.Generator): Generator object
            net (torch.nn.Module): Action value network
            beta (float, optional): hyperparameter for importance-sampling. Defaults to 0.4
            beta_flourish (float, optional): rate of increase in beta over time steps. Defaults to 0.001 
            eps_start (float, optional): Initial epsilon, probability of exploration. Defaults to 0.9.
            eps_decay (float, optional): Epsilon decay. Defaults to 1000.
            eps_end (float, optional): Final epsilon value, probability of exploration at the end of training. Defaults to 0.05.
            lr (float, optional): learning rate. Defaults to 0.01.
            discount (float, optional): discount factor, closer to 0 means agent values future rewards more. Defaults to 0.99.
            buffer_size (int, optional): size of experience buffer. Defaults to 10000.
            batch_size (int, optional): size experience batch to optimize model on. Defaults to 10000.
        """
        
        self.env = env
        self.target_net = copy.deepcopy(net).to(device)
        self.predict_net = net.to(device)
        self.optimizer = torch.optim.AdamW(
            self.predict_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.experiences = PrioritizedReplayBuffer(
            random=generator,
            buffer_size=buffer_size,
            batch_size=batch_size,
            alpha=0.6)
        self.writer = writer

        self.lr = lr
        self.beta = beta
        self.beta_flourish = beta_flourish
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.eps = 0

        self.discount = discount
        self.random = generator
        self.steps = 0
        self.device = device

    def policy(self, state: torch.Tensor):
        """Get the state and return action based on greedy epsilon policy

        Args:
            state (torch.Tensor): current state

        Returns:
            int: index of action
        """
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
        """optimize action value network
        """
        if len(self.experiences) < self.experiences.batch_size:
            return

        beta = 1 - (1 - self.beta) * np.exp(-1. * \
                    (self.steps - self.batch_size) * self.beta_flourish)

        batch, indices, weights = self.experiences.sample(beta)

        states, actions, rewards, next_states, dones = map(
            torch.cat, zip(*batch))

        self.predict_net.train()

        # Current Q values
        q_current = self.predict_net(states).gather(1, actions)

        # Next Q values based on target network
        with torch.no_grad():
            q_next = self.target_net(next_states).detach()  # 128 x 2

            # Double Q Learning, fix overestimation bias of q_value
            next_act = torch.argmax(self.predict_net(
                next_states).detach(), dim=1, keepdim=True)
            q_next = q_next.gather(1, next_act)

            q_next[dones] = 0.0
        q_target = rewards + (self.discount * q_next)

        td_error = torch.abs(q_target - q_current).squeeze()
        
        # prevent from zero out priority (proportional method)
        new_priorities = td_error + 0.01
        self.experiences.update_priorities(indices, new_priorities)
        weights = torch.tensor(weights).to(self.device, dtype=torch.float)

        # Compute loss and update model
        criterion = torch.nn.MSELoss()
        loss = criterion(weights * q_current, weights * q_target)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def train(self, episodes, tau):
        """training loop for agent

        Args:
            episodes (int): number of episodes to train
            tau (int): rate of target network update
        """
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
                state_tensor = torch.tensor(state).unsqueeze(
                    0).to(self.device, dtype=torch.float)
                next_state_tensor = torch.tensor(next_state).unsqueeze(
                    0).to(self.device, dtype=torch.float)
                action_tensor = torch.tensor(
                    [action]).unsqueeze(0).to(
                    self.device, dtype=torch.long)
                reward_tensor = torch.tensor(
                    [reward]).unsqueeze(0).to(
                    self.device, dtype=torch.float)
                done_tensor = torch.tensor([done]).unsqueeze(0).to(self.device)
                self.experiences.add(
                    state_tensor,
                    action_tensor,
                    reward_tensor,
                    next_state_tensor,
                    done_tensor)

                state = next_state
                self.optimize_model()
            if episode % tau == 0:
                self.target_net.load_state_dict(self.predict_net.state_dict())

            self.writer.add_scalar(
                'Performance/total reward over episodes', total_reward, episode)
    
    
    def eval(self, episodes=10, verbose=False):
        """evaluation method. Run the agent under current network for certain number of episodes and get the average reward

        Args:
            episodes (int, optional): Number of episodes for evaluation. Defaults to 10.
            verbose (bool, optional): Defaults to False.

        Returns:
            mean_reward: average reward over episodes
        """
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
    # Experiment setup
    exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')  # Unique experiment name based on current timestamp
    gym_id = 'CartPole-v1'  # Environment ID for Gym
    seed = 1  # Seed for reproducibility
    max_episodes = 1500  # Maximum number of episodes

    # Training and evaluation parameters
    device = torch.device('cpu')  # Device for training (CPU or CUDA)
    capture_video = True  # Flag to determine whether to capture videos
    video_record_freq = 100  # Frequency of recording video episodes
    eval_episodes = 50  # Number of episodes for evaluation

    # SAC hyperparameters
    discount = 0.99  # Discount factor for future rewards
    lr = 1e-4  # Learning rate
    beta = 0.4  # Beta parameter for SAC
    beta_flourish = 0.001  # Beta flourish parameter for SAC
    eps_start = 1  # Initial epsilon for exploration
    eps_decay = 1000  # Epsilon decay rate
    eps_end = 0.05  # Final epsilon value for exploration
    buffer_size = 10000  # Size of the replay buffer
    batch_size = 128  # Size of minibatches for training
    tau = 100  # Target network update rate

    # Weights & Biases configuration (for experiment tracking)
    wandb_track = False  # Flag to enable/disable Weights & Biases tracking
    wandb_project_name = 'LunarLander'  # Project name in Weights & Biases
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
    
    logpath = f'./runs/RainbowDQN/{gym_id}/{exp_name}' 
    
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
    
    # Hyperparameters
    hparams = {
        'algorithm': 'RainbowDQN',
        'environment_id': gym_id,
        'random_seed': seed,
        'maximum_episodes': max_episodes,
        'learning_rate': lr,
        'beta': beta,
        'beta_flourish': beta_flourish,
        'epsilon_start': eps_start,
        'epsilon_decay': eps_decay,
        'epsilon_end': eps_end,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'tau': tau,
        'device': device,
    }
    
    # Initialize the Agent with specific hyperparameters
    agent = Agent(
        writer=writer,
        generator=generator,
        env=env,
        net=Net(np.prod(*env.observation_space.shape), env.action_space.n, [128, 128]),
        beta=beta,
        beta_flourish=beta_flourish,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        lr=lr,
        discount=discount,
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device
    )

    # Train the agent
    agent.train(max_episodes, tau)


    # Evaluation and save metrics
    metrics = {
        'reward_eval': agent.eval(episodes=eval_episodes)
    }

    writer.add_hparams(hparams, metrics)
    
    # Close the environment if necessary
    env.close()
