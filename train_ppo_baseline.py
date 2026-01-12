import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

def make_env(log_path):
    env = gym.make("Humanoid-v4")
    env = Monitor(env, filename=log_path)  # logs episode reward/length
    return env

if __name__ == "__main__":
    env = make_env("logs_baseline")
    
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

    # Fto test (e.g. 200k), increase later.
    model.learn(total_timesteps=200_000)
    model.save("ppo_humanoid_baseline")
    env.close()