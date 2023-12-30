import torch
import copy
from utils import ReplayBuffer, argmax
import gymnasium as gym
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from Network import Net

class Agent():
    def __init__(self, writer, device, env: gym.Env, generator :np.random.Generator, net: torch.nn.Module, eps_start :float = 0.9, eps_decay: float = 1000, eps_end :float = 0.05, lr: float = 0.01, discount :float =0.99,buffer_size:int = 10000, batch_size:int = 10000) -> None:
        self.env = env
        self.target_net = copy.deepcopy(net).to(device)
        self.predict_net = net.to(device)
        self.optimizer = torch.optim.AdamW(self.predict_net.parameters(), lr=lr)
        self.experiences = ReplayBuffer(random=generator, buffer_size=buffer_size, batch_size=batch_size)
        self.writer = writer
        
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.eps = 0
        
        self.discount = discount
        self.random = generator
        self.steps = 0
        self.device=device
    
    def policy(self, state: torch.Tensor):
        self.steps += 1
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps / self.eps_decay)
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
            q_next = self.target_net(next_states).detach() #128 x 2

            # Double DQN
            next_act = torch.argmax(self.predict_net(next_states).detach(), dim=1, keepdim=True)
            q_next = q_next.gather(1, next_act)
            
            q_next[dones] = 0.0
        q_target = rewards + (self.discount * q_next)

        # Compute loss and update model
        criterion = torch.nn.MSELoss()
        loss = criterion(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.writer.add_scalar('loss/steps', loss, self.steps)
        self.optimizer.step()
    
    def train(self, episodes, tau):
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.policy(torch.tensor(state).to(self.device))
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                total_reward += reward

                state_tensor = state
                next_state_tensor = next_state
                action_tensor = [action]
                reward_tensor = [reward]
                terminated_tensor = [done]

                self.experiences.add(state_tensor, action_tensor, reward_tensor, next_state_tensor, terminated_tensor)

                state = next_state
                self.optimize_model()

            if episode % tau == 0:
                self.target_net.load_state_dict(self.predict_net.state_dict())

            print(f"Episode {episode}, Total Reward: {total_reward}, eps:  {self.eps}")
            self.writer.add_scalar('total reward/episode', total_reward, episode)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser("VanillaDQN.py")
    parser.add_argument("num_episodes", help="Number of episodes to run.", type=int)
    parser.add_argument("tau", help="frequency to update target (every tau episodes)", type=int)
    args = parser.parse_args()
    
    env = gym.make("CartPole-v1", render_mode="human")
    action_space = env.action_space
    env.reset()
    obs_space = env.observation_space
    reward_range = env.reward_range
    generator, seed = gym.utils.seeding.np_random(0)
    writer = SummaryWriter()
    mps_device = torch.device('mps')
        
    net = Net(np.prod(*obs_space.shape), action_space.n, [128,128])
    agent = Agent(
        env=env,
        net=net,
        writer=writer,
        discount=0.99,
        eps_start=0.9,
        eps_decay = 5e3,
        eps_end=0.05,
        batch_size=64,
        lr=5e-5,
        generator = generator,
        device = mps_device
    )
    
    agent.train(args.num_episodes, args.tau)
    
    env.close()