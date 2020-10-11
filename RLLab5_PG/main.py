# %%
# All dependencies
from torch.optim import optimizer
from runners import Runner
from sampling import sample_episode
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
import matplotlib.pyplot as plt
import sys
import gym
import time

# Local dependencies
import plots
from losses import compute_reinforce_loss
from policies import NNPolicy
from sampling import sample_episode

# TQDM wrapper
def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

# Define a policy and environment
policy = NNPolicy()
env = gym.envs.make("CartPole-v1")

# Define the optimizer we wish to use
optimizer = optim.Adam(policy.parameters())

# Define the runner we use to run this experiment
runner = Runner(env, policy)

# %%
nr_eps = 500
discount_rate = 1.0

runner.run(
    optimizer=optimizer, 
    num_episodes=nr_eps, 
    discount_factor=discount_rate, 
    sampling_function=sample_episode, 
    loss_fn=compute_reinforce_loss
)

plots.plot_durations(runner.episode_durations)
