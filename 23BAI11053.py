import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("BipedalWalker-v3")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)

model.save("models/ppo_bipedal")