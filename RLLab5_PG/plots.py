import numpy as np
import matplotlib.pyplot as plt

#%% Smoothing function for nicer plots
def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_durations(episode_durations):
    plt.plot(smooth(episode_durations, 10))
    plt.title('Episode durations per episode')
    plt.legend(['Policy gradient'])
