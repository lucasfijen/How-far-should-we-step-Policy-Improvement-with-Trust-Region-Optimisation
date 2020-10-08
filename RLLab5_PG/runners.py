from typing import Callable

from policies import NNPolicy
import gym
import torch
import torch.optim as optim

class Runner:
    def __init__(self, env: gym.Env, policy: NNPolicy) -> None:
        self.env = env
        self.policy = policy

    def run(
        self,
        optimizer, 
        num_episodes, 
        discount_factor, 
        sampling_function, 
        loss_fn,
        runner_fn: Callable = None
    ):
        runner_fn = self.default_runner_fn if runner_fn is None else runner_fn

        runner_fn(optimizer, self.policy, self.env, num_episodes, discount_factor, sampling_function, loss_fn)


    # TODO: can these follow some type
    def default_runner_fn(
        self,
        optimizer: optim.Optimizer,
        policy: NNPolicy,
        env: gym.Env,
        num_episodes: int, 
        discount_factor: float, 
        sampling_function: Callable, 
        loss_fn: Callable
    ):
        episode_durations = []

        for i in range(num_episodes):

            with torch.no_grad():
                episode = sampling_function(env, policy)

            loss = loss_fn(policy, episode, discount_factor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                           
            if i % 10 == 0:
                print("{2} Episode {0} finished after {1} steps"
                    .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
            episode_durations.append(len(episode[0]))
        
        return episode_durations
