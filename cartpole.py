import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")

# Create the model
model = PPO("MlpPolicy", env, verbose=1)

# Train it
model.learn(total_timesteps=10000)

# Watch it perform
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()