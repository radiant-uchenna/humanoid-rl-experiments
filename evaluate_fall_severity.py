import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import the correct fall-aware wrapper
from train_ppo_fallaware import FallAwareWrapper


def eval_model(model, env, n_episodes=20):
    min_heights = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        min_torso = np.inf

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Extract torso height from info (set by wrapper)
            torso_h = info.get("torso_height", None)

            if torso_h is not None and np.isfinite(torso_h):
                if torso_h < min_torso:
                    min_torso = torso_h

        min_heights.append(min_torso)

    return np.array(min_heights)

def record_episode_data(model, env):
    """
    Runs ONE deterministic episode and records:
    - torso heights over time
    - action magnitudes over time
    """
    obs, info = env.reset()
    done = False

    torso_heights = []
    action_magnitudes = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # torso height from wrapper
        torso_h = info.get("torso_height", None)
        if torso_h is not None:
            torso_heights.append(torso_h)

        # action magnitude (L2 norm)
        action_magnitudes.append(np.linalg.norm(action))

    return np.array(torso_heights), np.array(action_magnitudes)


def plot_additional_visuals(
    torso_base, torso_fall, act_base, act_fall
):
    """
    Generates:
    A. Torso height over time
    B. Action magnitude histogram
    """

    # -----------------------------
    # A. Torso height over time
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(torso_base, label="Baseline PPO")
    plt.plot(torso_fall, label="Fall-aware PPO")
    plt.xlabel("Timestep")
    plt.ylabel("Torso height")
    plt.title("Torso height over time (single episode)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("torso_height_over_time.png", dpi=300)
    plt.show()

    # -----------------------------
    # B. Action magnitude histogram
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(act_base, bins=20, alpha=0.6, label="Baseline PPO")
    plt.hist(act_fall, bins=20, alpha=0.6, label="Fall-aware PPO")
    plt.xlabel("Action magnitude (L2 norm)")
    plt.ylabel("Count")
    plt.title("Action magnitude distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("action_magnitude_hist.png", dpi=300)
    plt.show()

def collect_action_magnitudes(model, env, n_episodes=20):
    all_mags = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            all_mags.append(np.linalg.norm(action))

    return np.array(all_mags)


if __name__ == "__main__":
    # -----------------------------
    # Baseline model (no wrapper)
    # -----------------------------
    print("Loading baseline model...")
    env_base = gym.make("Humanoid-v4")
    env_base = FallAwareWrapper(env_base, fall_height_threshold=0.0, penalty_scale=0.0)
    model_base = PPO.load("ppo_humanoid_baseline")

    print("Evaluating baseline model...")
    base_heights = eval_model(model_base, env_base, n_episodes=20)

    # -----------------------------
    # Fall-aware model (with wrapper)
    # -----------------------------
    print("\nLoading fall-aware model...")
    env_fall = gym.make("Humanoid-v4")
    env_fall = FallAwareWrapper(env_fall)  # wrapper adds torso height + penalty
    model_fall = PPO.load("ppo_humanoid_fallaware")

    print("Evaluating fall-aware model...")
    fall_heights = eval_model(model_fall, env_fall, n_episodes=20)

     # Collect action magnitudes over multiple episodes
    act_base_all = collect_action_magnitudes(model_base, env_base, n_episodes=20)
    act_fall_all = collect_action_magnitudes(model_fall, env_fall, n_episodes=20)

    # -----------------------------
    # Filter invalid values
    # -----------------------------
    base_heights = base_heights[np.isfinite(base_heights)]
    fall_heights = fall_heights[np.isfinite(fall_heights)]

    print("\nBaseline heights:", base_heights)
    print("Fall-aware heights:", fall_heights)

    # -----------------------------
    # Plot histogram
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(base_heights, bins=10, alpha=0.6, label="Baseline PPO")
    plt.hist(fall_heights, bins=10, alpha=0.6, label="Fall-aware PPO")
    plt.xlabel("Minimum torso height during episode")
    plt.ylabel("Count")
    plt.title("Fall severity comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fall_severity_hist.png", dpi=300)
    plt.show()

        # -----------------------------
    # Additional visualisations
    # -----------------------------
    print("\nRecording single-episode data for additional plots...")

    torso_base, act_base = record_episode_data(model_base, env_base)
    torso_fall, act_fall = record_episode_data(model_fall, env_fall)

    plot_additional_visuals(
        torso_base, torso_fall,
        act_base, act_fall
    )
        # Action magnitude histogram over many episodes
    plt.figure(figsize=(8, 5))
    plt.hist(act_base_all, bins=20, alpha=0.6, label="Baseline PPO")
    plt.hist(act_fall_all, bins=20, alpha=0.6, label="Fall-aware PPO")
    plt.xlabel("Action magnitude (L2 norm)")
    plt.ylabel("Count")
    plt.title("Action magnitude distribution across episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig("action_magnitude_hist.png", dpi=300)
    plt.show()

