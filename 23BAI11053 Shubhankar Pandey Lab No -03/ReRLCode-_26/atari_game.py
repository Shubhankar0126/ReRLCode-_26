import gymnasium as gym
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make("PongNoFrameskip-v4", render_mode="human")
env = AtariWrapper(env)

env = DummyVecEnv([lambda: env])

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./pong_dqn/"
)


model.learn(total_timesteps=100000)

model.save("pong_dqn")

obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)