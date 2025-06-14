import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def load_rewards(pickle_path='statistics.pkl'):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File '{pickle_path}' not found.")

    with open(pickle_path, 'rb') as f:
        rewards = pickle.load(f)

    return rewards

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_rewards(rewards, save_path=None):
    plt.figure(figsize=(12, 6))
    
    # Plot total reward per episode
    plt.plot(rewards, label='Total Reward', color='blue', alpha=0.6)
    
    # Plot moving average
    ma = moving_average(rewards, window_size=100)
    plt.plot(range(99, len(rewards)), ma, label='100-Episode Moving Average', color='red', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    rewards = load_rewards('statistics.pkl')
    plot_rewards(rewards)
