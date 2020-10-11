from typing import Callable, Union

from policies import NNPolicy
import gym
import torch
import torch.optim as optim

class Runner:
    def __init__(self, env: gym.Env, policy: NNPolicy) -> None:
        self.env = env
        self.policy = policy
        self.episode_durations = []

    def run(
        self,
        optimizer, 
        num_episodes, 
        discount_factor, 
        sampling_function, 
        loss_fn,
    ):
        """Perform a run, and store latest results in `self.episode_durations`"""

        episode_durations = []

        for i in range(num_episodes):
            # Clean up old gradients first
            optimizer.zero_grad()

            # First sample the entire episode
            with torch.no_grad():
                episode = sampling_function(self.env, self.policy)
            
            # Calculate the entire loss, and store gradients of parameters
            loss = loss_fn(self.policy, episode, discount_factor)
            loss.backward()
            
            # Use gradients of parameters to improve
            optimizer.step()
            
            # Every 10 iterations, print results
            if i % 10 == 0:
                print("{2} Episode {0} finished after {1} steps".format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))

            episode_durations.append(len(episode[0]))
        
        self.episode_durations = episode_durations
        return episode_durations
