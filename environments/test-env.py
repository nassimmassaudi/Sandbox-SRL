import gym
import cv2
import numpy as np
import os

# Create the environment
env = gym.make('DeepmindLabNavEnv-v0', level='nav_maze_static_01')  # Adjust the level as needed

# Directory to save images or videos
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Initialize the environment and get the first observation
obs = env.reset()

# Number of steps to simulate
num_steps = 100

# Initialize a list to store video frames
video_frames = []

# Simulate the environment with random actions
for step in range(num_steps):
    # Choose a random action from the action space
    action = env.action_space.sample()
    
    # Step through the environment
    obs, reward, done, info = env.step(action)
    
    # Save the observation as an image
    image_path = os.path.join(output_dir, f"frame_{step:03}.png")
    cv2.imwrite(image_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    
    # Add the observation to the video frames
    video_frames.append(obs)
    
    if done:
        # Reset the environment if done
        env.reset()

# Create a simple video from the stored frames
video_path = os.path.join(output_dir, "simulation.avi")
height, width, _ = video_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

# Write each frame to the video
for frame in video_frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

out.release()

print(f"Simulation completed. Video saved to {video_path}. Images saved to {output_dir}/")
