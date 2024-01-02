"""
Author: Minh Pham-Dinh
Created: Jan 1st, 2024
Last Modified: Jan 1st, 2024
Email: mhpham26@colby.edu

Description:
    Implementation of Rainbow Deep Q Network.
    
    The implementation is based on:
    M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning," 
    arXiv preprint arXiv:1710.02298, Oct. 2017. 
    Available: https://arxiv.org/abs/1710.02298
"""

import torch
import copy
from utils import ReplayBuffer, PrioritizedReplayBuffer, argmax, parse_hyperparameters
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
                'total reward over episodes', total_reward, episode)
        torch.save(
            self.predict_net.state_dict(),
            f'./runs/RainbowDQN/tau_{tau}_batch_{self.batch_size}_epsdecay_{self.eps_decay}_lr_{self.lr}/saved_weights')

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
    parser = argparse.ArgumentParser("RainbowDQN.py")
    parser.add_argument(
        "--num_episodes",
        help="Number of episodes to run.",
        type=int,
        default=1000)
    parser.add_argument(
        "--tau",
        help="Frequency to update target (every tau episodes)",
        type=int,
        default=1000)
    parser.add_argument(
        "--learning_rate",
        help="Learning rate for the optimizer.",
        type=float,
        default=0.001)
    parser.add_argument(
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=64)
    parser.add_argument(
        "--epsilon_decay",
        help="Rate at which epsilon decays.",
        type=float,
        default=1e-5)
    parser.add_argument(
        "--eval_episodes",
        help="Number of episodes for evaluation.",
        type=int,
        default=100)
    parser.add_argument(
        "--grid_search",
        help="Perform grid search.",
        action='store_true')
    parser.add_argument(
        "--eval_mode",
        help="Perform evaluation mode with pretrained.",
        action='store_true')
    parser.add_argument("--cuda", help="use cuda.", action='store_true')
    parser.add_argument("--mps", help="use mps (Mac M1)", action='store_true')
    args = parser.parse_args()

    if args.cuda:
        device = torch.device('cuda')
    elif args.mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if args.eval_mode:
        env = gym.make("LunarLander-v2", render_mode="human")
        action_space = env.action_space
        env.reset()
        obs_space = env.observation_space
        filepath = input('path to trained model file: ')
        trained_net = Net(np.prod(*obs_space.shape),
                          action_space.n, [128, 128])
        trained_net.load_state_dict(
            torch.load(
                filepath,
                map_location=torch.device('cpu')))
        env.reset()
        obs_space = env.observation_space
        generator, seed = gym.utils.seeding.np_random(0)
        trained_agent = Agent(
            env=env,
            writer=None,
            eps_start=0,
            eps_end=0,
            device=device,
            generator=generator,
            net=trained_net
        )

        trained_agent.eval(args.eval_episodes)
        env.close()

    else:
        # env = gym.make("CartPole-v1", render_mode="human")
        env = gym.make("LunarLander-v2", render_mode="human")
        action_space = env.action_space
        env.reset()
        obs_space = env.observation_space
        generator, seed = gym.utils.seeding.np_random(0)

        if args.grid_search:
            filepath = input('link to hyperparameters range file: ')
            hyperparas = parse_hyperparameters(filepath)
            learning_rates = hyperparas['learning_rate']
            batch_sizes = hyperparas['batch_size']
            taus = hyperparas['tau']
            epsilon_decays = hyperparas['epsilon_decay']

            print(taus, batch_sizes, epsilon_decays, learning_rates)
            # Grid search
            for tau, batch_size, eps_decay, lr in itertools.product(
                    taus, batch_sizes, epsilon_decays, learning_rates):
                print(f'tau_{tau}_batch_{batch_size}_epsdecay_{eps_decay}')

                # Initialize writer for each combination
                writer = SummaryWriter(
                    log_dir=f'runs/RainbowDQN/tau_{tau}_batch_{int(batch_size)}_epsdecay_{eps_decay}_lr_{lr}')

                # Initialize the Agent with specific hyperparameters
                agent = Agent(
                    env=env,
                    net=Net(np.prod(*obs_space.shape), action_space.n, [128, 128]),
                    writer=writer,
                    discount=0.99,
                    eps_start=1,
                    eps_decay=eps_decay,
                    eps_end=0.01,
                    batch_size=int(batch_size),
                    lr=lr,
                    generator=generator,
                    device=device
                )

                # Train the agent
                agent.train(args.num_episodes, tau)

                # Close the environment if necessary
                env.close()

        else:
            writer = SummaryWriter(
                log_dir=f'runs/RainbowDQN/tau_{args.tau}_batch_{args.batch_size}_epsdecay_{args.epsilon_decay}_lr_{args.learning_rate}')
            # Initialize the Agent with specific hyperparameters
            agent = Agent(
                env=env,
                net=Net(np.prod(*obs_space.shape), action_space.n, [128, 128]),
                writer=writer,
                discount=0.99,
                eps_start=1,
                eps_decay=args.epsilon_decay,
                eps_end=0.01,
                batch_size=args.batch_size,
                lr=args.learning_rate,
                generator=generator,
                device=device
            )

            # Train the agent
            agent.train(args.num_episodes, args.tau)

            # Close the environment if necessary
            env.close()
