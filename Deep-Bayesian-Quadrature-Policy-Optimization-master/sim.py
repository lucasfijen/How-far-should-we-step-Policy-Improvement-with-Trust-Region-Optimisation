import torch
from torch.autograd import Variable
from running_state import ZFilter

def sim_episode(env, policy, max_episode_steps, result_writer):
    try:
        running_state = ZFilter((env.observation_space.shape[0], ), clip=5)

        state = env.reset()
        state = running_state(state)
        frames_store = []
        for t in range(max_episode_steps):  # Don't infinite loop while learning
            # Simulates one episode, i.e., until the agent reaches the terminal state or has taken 10000 steps in the environment
            action_mean, action_log_std, action_std = policy(
                Variable(torch.Tensor([state])))
            action = torch.normal(action_mean, action_std).detach().data[0].cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)

            state = next_state
            frames = env.render('rgb_array')
            frames_store.append(frames)
        
        return frames_store

    except Exception as e:
        print(f'Tried running simulation, but got error:{e}')

        return []