import gymnasium as gym
import numpy as np
import pygame
import time
import sys

def run_car_racing_manual():
    # Create the environment
    env = gym.make("CarRacing-v2", render_mode="human")
    
    # Reset the environment and get the initial observation
    observation, info = env.reset()
    
    # Initialize action
    action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]
    
    # Variables to track episode progress
    total_reward = 0
    steps = 0
    done = False
    truncated = False
    
    print("Controls:")
    print("  Arrow Up: Accelerate")
    print("  Arrow Down: Brake")
    print("  Arrow Left/Right: Steer")
    print("  Q: Quit")
    
    # Run the episode until it's done or truncated
    while not (done or truncated):
        # Process keyboard input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            
        keys = pygame.key.get_pressed()
        
        # Reset action
        action = np.array([0.0, 0.0, 0.0])
        
        # Steering (left/right arrows)
        if keys[pygame.K_LEFT]:
            action[0] = -1.0  # Steer left
        elif keys[pygame.K_RIGHT]:
            action[0] = 1.0   # Steer right
        
        # Gas (up arrow)
        if keys[pygame.K_UP]:
            action[1] = 1.0   # Apply gas
        
        # Brake (down arrow)
        if keys[pygame.K_DOWN]:
            action[2] = 1.0   # Apply brake
            
        # Quit (Q key)
        if keys[pygame.K_q]:
            break
        
        # Take a step in the environment
        observation, reward, done, truncated, info = env.step(action)
        
        # Update episode metrics
        total_reward += reward
        steps += 1
        
        # Add a small delay to make control easier
        time.sleep(0.01)
    
    print(f"Episode finished after {steps} steps with total reward {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    run_car_racing_manual()