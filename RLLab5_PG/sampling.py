import torch

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
