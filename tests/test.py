import gymnasium as gym
import ale_py
import miniworld

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

print(gym.envs.registry.keys())

# env = gym.make("MiniWorld-OneRoom-v0")
# env = gym.make('Breakout-ramNoFrameskip-v4')
# env = gym.make('CartPole-v1')

# import sys
# import os

# script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
# parent_dir = os.path.dirname(script_dir)  # Get the parent directory

# print("script_dir :", script_dir)
# print("parent_dir :", parent_dir)

# import jax
# print("Devices:", jax.devices())


# import shimmy
# print(shimmy.__version__)
