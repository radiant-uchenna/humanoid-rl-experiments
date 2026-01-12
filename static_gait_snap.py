import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("Humanoid-v4", render_mode="rgb_array")
model = PPO.load("ppo_humanoid_baseline")  # update path if needed

TARGET_EPISODE = 3900
episode_count = 0

# Run episodes until we reach episode 2500
while episode_count < TARGET_EPISODE:
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    episode_count += 1

# Now we are at episode 2500 — capture frames from THIS episode
obs, _ = env.reset()
frames = []

max_steps = 1000  # safety cap

for step in range(max_steps):
    frame = env.render()
    frames.append(frame)

    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        break

env.close()

# Pick 5 evenly spaced frames
num_frames = len(frames)
indices = np.linspace(0, num_frames - 1, 5, dtype=int)

for i, idx in enumerate(indices):
    frame = frames[idx]
    plt.figure(figsize=(4, 4))
    plt.imshow(frame)
    plt.axis("off")
    plt.title(f"Episode 3900 — step {idx}")
    plt.savefig(f"snapshot_ep3900_{i}.png", dpi=300)
    plt.close()
