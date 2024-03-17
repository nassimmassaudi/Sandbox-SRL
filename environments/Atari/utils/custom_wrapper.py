import gym
import numpy as np
import os
import cv2

class GaussianNoiseWrapper(gym.ObservationWrapper):
    """
    Wrapping adding Gaussian Noise to the background of Atari Games
    """
    def __init__(self, env, mean=0.0, std=1.0):
        super().__init__(env)
        self.mean = mean
        self.std = std

    def observation(self, observation):
        noise = np.random.normal(self.mean, self.std, observation.shape).astype(np.uint8)
        noisy_obs = observation + noise
        # Clip the observations to maintain the original observation's range
        noisy_obs = np.clip(noisy_obs, 0, 255)
        return noisy_obs



class GaussianNoiseWrapperWithRecording(gym.ObservationWrapper):
    def __init__(self, env, mean=0.0, std=1.0, record_path=None):
        super().__init__(env)
        self.mean = mean
        self.std = std
        self.record_path = record_path
        if record_path and not os.path.exists(record_path):
            os.makedirs(record_path)
        self.frame_count = 0

    def observation(self, observation):
        # Apply Gaussian noise
        noisy_obs = super().observation(observation)
        # Save frame if recording is enabled
        if self.record_path:
            frame_path = os.path.join(self.record_path, f"frame_{self.frame_count:08d}.png")
            cv2.imwrite(frame_path, noisy_obs)
            self.frame_count += 1
        return noisy_obs

    def reset(self, **kwargs):
        self.frame_count = 0  # Reset frame count on each episode
        return super().reset(**kwargs)
