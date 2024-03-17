# import gym
# # import ale_py

# # print('gym:', gym.__version__)
# # # print('ale_py:', ale_py.__version__)

# # env = gym.make('Breakout-v0')


# import sys
# import os

# script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
# parent_dir = os.path.dirname(script_dir)  # Get the parent directory

# print("script_dir :", script_dir)
# print("parent_dir :", parent_dir)

import jax
print("Devices:", jax.devices())
