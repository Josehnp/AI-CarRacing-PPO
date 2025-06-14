import gymnasium as gym
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torch import nn

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing function (same as training)
def image_preprocessing(img):
    img = cv2.resize(img, dsize=(84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

# Environment wrapper (same as training)
class CarEnvironment(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, no_operation=50, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames

    def reset(self):
        observation, info = self.env.reset()
        for _ in range(self._no_operation):
            observation, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                break
        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

    def step(self, action):
        total_reward = 0
        for _ in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        observation = image_preprocessing(observation)
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        return self.stack_state, total_reward, terminated, truncated, info

# Define the same Actor network used during training
class Actor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._n_features = 32 * 9 * 9
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view((-1, self._n_features))
        x = self.fc(x)
        return x

# Load model
actor = Actor(in_channels=4, out_channels=5).to(device)
actor.load_state_dict(torch.load("best_actor.pth", map_location=device))
actor.eval()

# Create environment
env = gym.make('CarRacing-v2', continuous=False, render_mode='human')
env = CarEnvironment(env)

# Run one episode
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_logits = actor(obs_tensor)
        action = torch.argmax(action_logits, dim=-1).item()

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print(f"Total reward: {total_reward:.2f}")
