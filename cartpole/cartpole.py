import gymnasium as gym
from stable_baselines3 import PPO
import os

# Create the environment (from OpenAI's Gymnasium library)
env = gym.make("CartPole-v1", render_mode="human")

# Create the model using Rl agent PPO
model = PPO("MlpPolicy", env, verbose=1)

# Train longer
model.learn(total_timesteps=50000)

# Save the model at models directory
os.makedirs("models", exist_ok=True)
model.save("models/cartpole_ppo")
print("Model saved!")

# Watch it perform for longer
# As it fails, PPO learns what happens and changes MlpPolicy slightly
# episode stands for what attempt its on, 
# episode_reward stands for how many steps it was able to keep the pole up for
# end = "\r" changes the counter each time instead of printing new outputs everytime it fails
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
# Run on wsl terminal using python cartpole.py
env.close()