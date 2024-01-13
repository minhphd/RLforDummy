import gymnasium as gym

gym_id = "HalfCheetah-v4"

env = gym.make(gym_id, render_mode="human")

obs, _ = env.reset()

print(env.action_space)
print(env.observation_space)
    
torch.nn