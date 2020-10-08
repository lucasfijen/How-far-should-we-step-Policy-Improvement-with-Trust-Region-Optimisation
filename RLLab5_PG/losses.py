import torch

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