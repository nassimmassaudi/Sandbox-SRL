import os
from gym.envs.registration import register

# Register a standard Deepmind Lab environment. Users need to pass the appropriate arguments when using it.
register(
    id='DeepmindLabEnv-v0',
    entry_point='DMLab.env:DeepmindLabEnvironment',
    kwargs={'level': 'seekavoid_arena_01', 'configs': {}, 'observation_keys': ['RGB_INTERLEAVED'], 'height': 84, 'width': 84, 'frame_skip': 4, 'fps': 60}
)

# Register a navigation-specific environment for Deepmind Lab.
register(
    id='DeepmindLabNavEnv-v0',
    entry_point='DMLab.env:DeepmindLabMazeNavigationEnvironment',
    kwargs={'level': 'nav_maze_random_goal_01', 'width': 84, 'height': 84, 'frame_skip': 4, 'fps': 60, 'enable_depth': False, 'other_configs': {}, 'other_obs': []}
)

# Register the continuous action space environment version of Deepmind Lab.
register(
    id='ContinuousDeepmindLabEnv-v0',
    entry_point='DMLab.env:ContinuousDeepmindLabEnvironment',
    kwargs={'level': 'seekavoid_arena_01', 'configs': {}, 'observation_keys': ['RGB_INTERLEAVED'], 'height': 84, 'width': 84, 'frame_skip': 1, 'fps': 60}
)

# import os
# from gym import register

# # Register the Deepmind Lab Environment
# # Default environment
# register(
#     id='DeepmindLabEnv-v0',
#     entry_point='DMLab.env:DeepmindLabEnvironment',
# )

# # Navigation environment
# register(
#     id='DeepmindLabNavEnv-v0',
#     entry_point='DMLab.env:DeepmindLabMazeNavigationEnvironment',
# )

# # Registering the continuous action space environment
# register(
#     id='ContinuousDeepmindLabEnv-v0',
#     entry_point='DMLab.env:ContinuousDeepmindLabEnvironment',  # Adjust based on the exact path
# )

