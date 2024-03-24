import gymnasium as gym
import os
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import numpy as np
import imageio

# Path to the current script
current_script_path = os.path.abspath(__file__)

# Path to the directory containing the current script
current_dir = os.path.dirname(current_script_path)

# Path to the videos directory
videos_dir = os.path.join(current_dir, "videos")


# def save_frames_as_images(obs, frame_indices, output_dir=f"{videos_dir}/debug_frames"):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for idx in frame_indices:
#         frame_path = os.path.join(output_dir, f"frame_{idx}.png")
#         cv2.imwrite(frame_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        
        
class CustomVideoRecorder(gym.Wrapper):
    def __init__(self, env, save_path):
        super().__init__(env)
        self.save_path = save_path
        self.frames = []

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.frames = [self.env.render()]
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.frames.append(self.env.render())
        if terminated or truncated:
            self.save_video()
            self.frames = []
        return observation, reward, terminated, truncated, info

    def save_video(self):
        with imageio.get_writer(self.save_path, fps=30) as video:
            for frame in self.frames:
                video.append_data(frame)


class SparseRandomColorBlackPixelsWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_density=0.5):
        super(SparseRandomColorBlackPixelsWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), "This wrapper only works with continuous observation spaces (Box)"
        self.observation_space = env.observation_space
        self.noise_density = noise_density
    
    def observation(self, obs):
        # Identify black pixels (where all RGB values are 0)
        black_pixels = np.all(obs == [0,0,0], axis=-1)
    
        # Calculate number of black pixels to randomly color based on density
        num_black_pixels = np.sum(black_pixels)
        num_pixels_to_color = int(num_black_pixels * self.noise_density)
        
        # Choose random black pixels to color
        black_pixels_indices = np.argwhere(black_pixels)
        indices_to_color = black_pixels_indices[np.random.choice(black_pixels_indices.shape[0], size=num_pixels_to_color, replace=False)]
        
        # Generate random colors for these pixels
        random_colors = np.random.randint(0, 256, size=(indices_to_color.shape[0], 3), dtype=np.uint8)
        
        # Replace selected black pixels with random colors
        for idx, color in zip(indices_to_color, random_colors):
            obs[tuple(idx)] = color
        
        return obs


env_id = "BreakoutNoFrameskip-v4"

env = gym.make(env_id, render_mode="rgb_array")
env = SparseRandomColorBlackPixelsWrapper(env, noise_density=0.05)
env = CustomVideoRecorder(env, f"{videos_dir}/{env_id}")


env.reset()
for k in range(800):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    # save_frames_as_images(observation, [0, 1, 2])

    if terminated or truncated:
        print(f"Finished episode in {k} timesteps.")
        break
