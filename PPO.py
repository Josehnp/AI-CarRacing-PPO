import gymnasium as gym
import matplotlib
from matplotlib.animation import FuncAnimation
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import random
import math
import pickle
from torch.optim import Adam
import time
import os
import glob
from datetime import datetime


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_preprocessing(img):
  img = cv2.resize(img, dsize=(84, 84))
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
  return img

class CarEnvironment(gym.Wrapper):
  def __init__(self, env, skip_frames=4, stack_frames=4, no_operation=50, **kwargs):
    super().__init__(env, **kwargs)
    self._no_operation = no_operation
    self._skip_frames = skip_frames
    self._stack_frames = stack_frames

  def reset(self):
    observation, info = self.env.reset()

    for i in range(self._no_operation):
      observation, reward, terminated, truncated, info = self.env.step(0)

    observation = image_preprocessing(observation)
    self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
    return self.stack_state, info


  def step(self, action):
    total_reward = 0
    for i in range(self._skip_frames):
      observation, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward

      if terminated or truncated:
        break

    observation = image_preprocessing(observation)
    self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
    return self.stack_state, total_reward, terminated, truncated, info

class Actor(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs):
    super().__init__(*args, **kwargs)
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



class Critic(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs):
    super().__init__(*args, **kwargs)
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

class PPO:
  def __init__(self, action_dim=5, obs_dim=4, trajectories=512, gamma=0.99, lr_actor=3e-4, lr_critic=1e-3, clip=0.2, n_updates=10, lambda_=0.99,
               moving_avg_window=100, convergence_threshold=0.01, check_every=50, patience=10, max_episodes=100000):
    self.action_dim = action_dim
    self.obs_dim = obs_dim
    self.trajectories = trajectories
    self.gamma = gamma
    self.lr_actor = lr_actor
    self.lr_critic = lr_critic
    self.clip = clip
    self.n_updates = n_updates
    self.lambda_ = lambda_
    self._total_rewards = []
    self.actor = Actor(obs_dim, action_dim).to(device)
    self.critic = Critic(obs_dim, 1).to(device)
    self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)
    self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)
    
    # Convergence parameters
    self.moving_avg_window = moving_avg_window  
    self.convergence_threshold = convergence_threshold
    self.check_every = check_every             
    self.patience = patience                   
    self.max_episodes = max_episodes



  """
  This function takes as a parameter an observation, feeds it to the CNN and gets the
  raw predictions (logits) and it samples an action through the categorical distribution.
  It returns the action and the logarithmic probability.
  """
  def get_action(self, obs):
    # Our observation is a 2D numpy array so we first create a tensor
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    # Feeding the tensor to the actor and get the logits
    action_probs = self.actor(obs)

    # Creating a Categorical distribution
    dist = Categorical(logits=action_probs)

    # Sampling the action
    action = dist.sample()

    log_prob = dist.log_prob(action)

    return action.detach().cpu().numpy(), log_prob.detach()


  """
  This function is where we collect the trajectories (e.g. observations, rewards and other information)
  using the current policy. We run this until we collect the number of trajectories we set.
  It returns all the collected information.
  """
  def collect_trajectories(self):
    batch_obs = []
    batch_rewards = []
    batch_log_probs = []
    batch_next_obs = []
    batch_actions = []
    batch_dones = []
    t = 0

    # Creating the discrete environment and passing it through the our wrapper for the modification
    env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
    env = CarEnvironment(env)

    while True:

      # Reset environment
      obs, _ = env.reset()

      # Runs as many times as needed until we get the number of trajectories we want
      while True:

        # Append current state
        batch_obs.append(obs)

        # Choose an action
        a, log_prob = self.get_action(obs)

        # Append action
        batch_actions.append(a)

        # Append log prob
        batch_log_probs.append(log_prob)

        # Perform the action
        obs, rew, terminated, truncated, _ = env.step(a.item())

        # Append reward
        batch_rewards.append(rew)

        # Increase the number of T horizon
        t += 1

        # Check criterion for loop termination
        if terminated or truncated or t == self.trajectories:
          batch_dones.append(1)
          break
        else:
          batch_dones.append(0)

      # Check criterion for loop termination
      if t == self.trajectories:
        env.close()
        break

    self._total_rewards.append(sum(batch_rewards))

    # Convert to tensors
    batch_obs = np.array(batch_obs)
    batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
    batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
    batch_actions = torch.tensor(batch_actions, dtype=torch.long)

    # Reward Normalization
    batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-8)

    return batch_obs, batch_rewards, batch_log_probs, batch_actions, batch_dones


  """
  Computing the discounted reward sum based on the current V values with
  GAE (Generalized Advantage Estimation)
  """
  def compute_discounted_sum(self, batch_rewards, V, batch_dones):
    discounted_sum = []
    gae = 0
    zero = torch.tensor([0])
    V = torch.cat((V.cpu(), zero))

    for i in reversed(range(len(batch_rewards))):
      delta = batch_rewards[i] + self.gamma * V[i + 1] * (1 - batch_dones[i]) - V[i]
      gae = delta + self.gamma * self.lambda_ * gae * (1 - batch_dones[i])
      discounted_sum.insert(0, gae)

    return discounted_sum


  """
  Make the agent learn the environment
  """
  def train(self):
      convergence_counter = 0
      episode = 0
      best_avg_reward = float('-inf')
      training_start_time = time.time()
      recent_averages = []  # Keep track of recent moving averages

      while episode < self.max_episodes:
          episode += 1

          # Every 10 episodes, print rewards
          if episode % 10 == 0:
              current_avg = np.mean(self._total_rewards[-20:]) if len(self._total_rewards) >= 20 else float('-inf')
              print(f"Episode {episode} - Average Reward (last 20 episodes): {current_avg:.2f}")

          # Save periodically
          if episode % 500 == 0:
              print("Processed: ", episode)
              torch.save(self.actor.state_dict(), f'actor_weights_{episode}.pth')
              torch.save(self.critic.state_dict(), f'critic_weights_{episode}.pth')
              with open('statistics.pkl', 'wb') as f:
                  pickle.dump((self._total_rewards), f)

          # Collect trajectories and train as before
          batch_obs, batch_rewards, batch_log_probs, batch_actions, batch_dones = self.collect_trajectories()
          
          # Compute critic values
          V = self.critic(batch_obs.to(device)).squeeze()

          # Rest of the training logic remains the same until convergence checking
          discounted_sum = self.compute_discounted_sum(batch_rewards, V, batch_dones)
          discounted_sum = torch.tensor(discounted_sum, dtype=torch.float32)
          advantages = discounted_sum - V.detach().cpu()
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

          for update in range(self.n_updates):
              actions_probs = self.actor(batch_obs.to(device))
              action_log_probs = actions_probs.gather(1, batch_actions.to(device)).squeeze()
              ratios = torch.exp(action_log_probs - batch_log_probs.to(device)).cpu()
              surr1 = ratios * advantages
              surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
              loss = -torch.min(surr1, surr2).mean()

              self.actor_optim.zero_grad()
              loss.backward(retain_graph=True)
              self.actor_optim.step()

              V = self.critic(batch_obs.to(device)).squeeze()
              value_loss = nn.MSELoss()(V, discounted_sum.detach().to(device))

              self.critic_optim.zero_grad()
              value_loss.backward()
              self.critic_optim.step()

          # Check for convergence
          if episode > self.moving_avg_window and episode % self.check_every == 0:
              recent_rewards = self._total_rewards[-self.moving_avg_window:]
              current_avg = np.mean(recent_rewards)
              recent_averages.append(current_avg)
              
              # Save best model if we have a new best average
              if current_avg > best_avg_reward:
                  best_avg_reward = current_avg
                  torch.save(self.actor.state_dict(), 'best_actor.pth')
                  torch.save(self.critic.state_dict(), 'best_critic.pth')
              
              # Check for convergence only if we have enough data
              if len(recent_averages) >= 3:  # Need at least 3 points to check stability
                  # Calculate relative changes between consecutive averages
                  changes = [abs((recent_averages[i] - recent_averages[i-1]) / recent_averages[i-1]) 
                           for i in range(len(recent_averages)-2, len(recent_averages))]
                  
                  avg_change = np.mean(changes)
                  print(f"Convergence check - Current avg: {current_avg:.2f}, Average relative change: {avg_change:.2%}")
                  
                  # Check if changes are consistently small
                  if avg_change < self.convergence_threshold:
                      convergence_counter += 1
                      print(f"Convergence counter: {convergence_counter}/{self.patience}")
                      if convergence_counter >= self.patience:
                          print(f"Model has converged at episode {episode}! Training complete.")
                          print(f"Final average reward: {current_avg:.2f}")
                          self._save_final_data(episode, training_start_time)
                          return
                  else:
                      convergence_counter = 0
                      
                  # Keep only recent history to avoid old averages affecting convergence check
                  if len(recent_averages) > 10:
                      recent_averages = recent_averages[-10:]

      print(f"Reached maximum episodes ({self.max_episodes}). Training stopped.")
      self._save_final_data(episode, training_start_time)

  def _save_final_data(self, final_episode, start_time):
      """Save final training data and statistics"""
      training_time = time.time() - start_time
      
      # Save final model weights
      torch.save(self.actor.state_dict(), 'final_actor.pth')
      torch.save(self.critic.state_dict(), 'final_critic.pth')
      
      # Save all training data
      training_data = {
          'total_rewards': self._total_rewards,
          'final_episode': final_episode,
          'training_time': training_time,
          'convergence_params': {
              'moving_avg_window': self.moving_avg_window,
              'convergence_threshold': self.convergence_threshold,
              'check_every': self.check_every,
              'patience': self.patience
          },
          'final_average_reward': np.mean(self._total_rewards[-self.moving_avg_window:]) if self._total_rewards else None
      }
      
      # Save complete training statistics
      with open('final_statistics.pkl', 'wb') as f:
          pickle.dump(training_data, f)
      
      print("\nTraining Summary:")
      print(f"Total Episodes: {final_episode}")
      print(f"Training Time: {training_time/3600:.2f} hours")
      print(f"Final Average Reward: {training_data['final_average_reward']:.2f}")

model = PPO(max_episodes=100000)  # Only override max_episodes, use optimized defaults for the rest
model.train()

eval_env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
eval_env = CarEnvironment(eval_env)

frames = []
scores = 0
s, _ = eval_env.reset()

done, ret = False, 0

while not done:
    frames.append(eval_env.render())
    s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    a = torch.argmax(model.actor(s), dim=-1)
    discrete_action = a.item() % 5
    s_prime, r, terminated, truncated, info = eval_env.step(discrete_action)
    s = s_prime
    ret += r
    done = terminated or truncated
    if terminated:
      print(terminated)
scores += ret

def animate(imgs, video_name, _return=True):
    import cv2
    import os
    import string
    import random

    if video_name is None:
        video_name = ''.join(random.choice(string.ascii_letters) for i in range(18)) + '.webm'
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()

animate(frames, None)
