#%% [markdown]
# # Reinforcement Learning - Policy Gradient
# If you want to test/submit your solution **restart the kernel, run all cells and submit the pg_autograde.py file into codegrade.**

#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


#%% 
%matplotlib inline
import matplotlib.pyplot as plt
import sys

import gym
import time

assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

#%% [markdown]
# ---

# ## 3. Policy Gradient

# ### 3.1 Policy Network

# In order to implement policy gradient, we will first implement a class with a policy network. Although in general this does not have to be the case, we will use an architecture very similar to the Q-network that we used (two layers with ReLU activation for the hidden layer). Since we have discrete actions, our model will output one value per action, where each value represents the (normalized!) probability of selecting that action. *Use the softmax activation function.*

#%% 
class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        output = self.l1(x.float())
        output = F.relu(output)
        output = self.l2(output)
        output = F.softmax(output)
        return output
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        y = self.forward(obs)
        action_probs = y.gather(1, actions)

        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        y = self.forward(obs)
        action = torch.distributions.categorical.Categorical(y).sample()
        return int(action)
        
        


#%% 
num_hidden = 128
torch.manual_seed(1234)
policy = NNPolicy(num_hidden)

states = torch.rand(10, 4)
actions = torch.randint(low=0, high=2, size=(10,1))
print(actions)

# Does the outcome make sense?
forward_probs = policy.forward(states)
print(forward_probs)
assert forward_probs.shape == (10,2), "Output of forward has incorrect shape."
sampled_action = policy.sample_action(states[0])
assert sampled_action == 0 or sampled_action == 1, "Output of sample action is not 0 or 1"

action_probs = policy.get_probs(states, actions)
print(action_probs)
assert action_probs.shape == (10,1), "Output of get_probs has incorrect shape."


#%% [markdown]
# ### 3.2 Monte Carlo REINFORCE

# Now we will implement the *Monte Carlo* policy gradient algorithm. Remember that this means that we will estimate returns for states by sample episodes. Compared to DQN, this means that we do *not* perform an update step at every environment step, but only at the end of each episode. This means that we should generate an episode of data, compute the REINFORCE loss (which requires computing the returns) and then perform a gradient step.

# * You can use `torch.multinomial` to sample from a categorical distribution.
# * The REINFORCE loss is defined as $- \sum_t \log \pi_\theta(a_t|s_t) G_t$, which means that you should compute the (discounted) return $G_t$ for all $t$. Make sure that you do this in **linear time**, otherwise your algorithm will be very slow! Note the - (minus) since you want to maximize return while you want to minimize the loss.

# To help you, we wrote down signatures of a few helper functions. Start by implementing a sampling routine that samples a single episode (similarly to the one in Monte Carlo lab).

#%% 
def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # TAKEN FROM MC
    max_len_episode = 1000
    s = env.reset()

    for i in range(max_len_episode):

        states.append(s)
        action = int(policy.sample_action(torch.tensor([s, ], dtype=torch.float)))

        new_state, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        s = new_state
        if done:
            break

    return torch.tensor(states), torch.tensor(actions).view(-1,1), torch.tensor(rewards).view(-1,1), torch.tensor(dones).view(-1,1)

#%% 
# Let's sample some episodes
env = gym.envs.make("CartPole-v1")
num_hidden = 128
torch.manual_seed(1234)
policy = NNPolicy(num_hidden)
for episode in range(3):
    trajectory_data = sample_episode(env, policy)

#%% [markdown]
# Now implement loss computation and training loop of the algorithm.

#%% 
def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere

    states, actions, rewards, dones = episode
    probs = policy.get_probs(states, actions)

    Gs = []
    G = 0

    for i, reward in enumerate(list(rewards)[::-1]):
        G = (discount_factor * G) + reward
        Gs.append(G)

    Gs = torch.tensor(Gs[::-1]).view(-1, 1)
    loss = - torch.sum(torch.log(probs) * Gs)

    return loss


def run_episodes_policy_gradient(
    policy, 
    env, 
    num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):

        with torch.no_grad():
            episode = sampling_function(env, policy)

        loss = compute_reinforce_loss(policy, episode, discount_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                           
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations

#%% Smoothing function for nicer plots
def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#%%  Feel free to play around with the parameters!
num_episodes = 500
discount_factor = 0.99
learn_rate = 0.001
seed = 42
env = gym.envs.make("CartPole-v1")
torch.manual_seed(seed)
env.seed(seed)
policy = NNPolicy(num_hidden)

episode_durations_policy_gradient = run_episodes_policy_gradient(
    policy, env, num_episodes, discount_factor, learn_rate)

plt.plot(smooth(episode_durations_policy_gradient, 10))
plt.title('Episode durations per episode')
plt.legend(['Policy gradient'])

#%% 


#%% 

