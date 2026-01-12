import gymnasium as gym
import numpy as np

class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.printed = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.printed:
            print("\nOBSERVATION SHAPE:", obs.shape)
            print("FIRST 50 VALUES:", obs[:50])
            self.printed = True

        info["torso_height"] = None
        return obs, reward, terminated, truncated, info