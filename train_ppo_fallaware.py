import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class FallAwareWrapper(gym.Wrapper):
    """
    Adds a penalty when the torso height drops below a threshold.
    Uses the true MuJoCo qpos root z-coordinate for torso height.
    """
    def __init__(self, env, fall_height_threshold=0.8, penalty_scale=5.0):
        super().__init__(env)
        self.fall_height_threshold = fall_height_threshold
        self.penalty_scale = penalty_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Correct torso height extraction from MuJoCo qpos
        torso_height = float(self.env.unwrapped.data.qpos[2])

        # Compute fall penalty
        fall_penalty = 0.0
        if torso_height < self.fall_height_threshold:
            fall_penalty = -self.penalty_scale * (self.fall_height_threshold - torso_height)

        reward = reward + fall_penalty

        # Log for analysis
        info["torso_height"] = torso_height
        info["fall_penalty"] = fall_penalty

        return obs, reward, terminated, truncated, info


def make_env(log_path):
    env = gym.make("Humanoid-v4")
    env = FallAwareWrapper(env, fall_height_threshold=0.8, penalty_scale=5.0)
    env = Monitor(env, filename=log_path)
    return env


if __name__ == "__main__":
    env = make_env("logs_fallaware")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
    )

    # Train — you can increase timesteps tomorrow for better results
    model.learn(total_timesteps=200_000)

    model.save("ppo_humanoid_fallaware")
    env.close()