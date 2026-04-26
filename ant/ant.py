import gymnasium as gym
from stable_baselines3 import PPO
import os

# Create the environment
env = gym.make("Ant-v4", render_mode="human")

# Create the model
model = PPO("MlpPolicy", env, verbose=1)

# Train it
model.learn(total_timesteps=50000)

# Save the model
os.makedirs("ant/models", exist_ok=True)
model.save("ant/models/ant_ppo")
print("Model saved!")

# Watch it perform
obs, info = env.reset()
episode = 1
episode_reward = 0

for step in range(5000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    print(f"Step: {step + 1} | Episode: {episode} | Episode Reward: {episode_reward}", end="\r")
    if terminated or truncated:
        print(f"Step: {step + 1} | Episode: {episode} | Episode Reward: {episode_reward}")
        episode += 1
        episode_reward = 0
        obs, info = env.reset()

env.close()