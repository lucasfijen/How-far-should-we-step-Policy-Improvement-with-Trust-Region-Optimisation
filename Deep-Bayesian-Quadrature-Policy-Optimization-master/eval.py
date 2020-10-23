import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils
from pathlib import Path
import seaborn as sns

plt.style.use('seaborn-poster') #sets the size of the charts
plt.style.use('ggplot')
matplotlib.rcParams['font.family'] = "serif"
sns.set_context('poster')

path_to_results = 'temp_results'
path_to_plots = 'temp_results/plots'
path_to_total_results = 'temp_results/total.csv'
merge_single_runs = False

def get_graph_style(group):
    alpha = 1
    if group.iloc[0]['run_model'] == 'NPG':
        # Blueish
        color = '#B83280'
        linestyle = 'solid'
    elif group.iloc[0]['run_model'] == 'TRPO':
        # Redish
        color = '#FC8181'
        linestyle = 'dashdot'
    else:
        # Greenish
        color = '#68D391'
        linestyle = 'dashdot'
    
    return {'color': color, 'linestyle': linestyle, 'alpha': alpha}

def merge_all_single_runs(filter_unfinished: bool = False):
    """Merge all single-run csvs"""
    total_df = pd.DataFrame()
    results_path = Path(path_to_results)
    single_runs_in_subdirs_paths = list(results_path.rglob('run-*/*.csv'))
    single_runs_in_root_paths = list(results_path.rglob('run-*/*.csv'))
    all_single_runs = [*single_runs_in_root_paths, *single_runs_in_subdirs_paths]

    for run_path in all_single_runs:
        single_df = pd.read_csv(run_path)

        # Let's do a number of checks

        # We ensure that our dataframe goes up until epoch 700
        if (filter_unfinished and not 700 in single_df['epoch'].values): continue
        
        total_df = total_df.append(single_df, ignore_index=True)
    total_df.to_csv(path_to_total_results)
    return total_df

def plot_stepsize_per_env(all_runs_df, envs=[]):
    # take in only one particular step-size, namely 0.001
    all_runs_df = all_runs_df[(
        (all_runs_df['run_model'] == 'TRPO')
        | (all_runs_df['step_size'] == 0.001)
    )]

    for env_name in envs:
        env_runs_df = all_runs_df[all_runs['env'] == env_name]
        env_per_model = env_runs_df.groupby('run_model')

        for name, group in env_per_model:
            graph_style = get_graph_style(group)

            # For each model, get all seeds, and sort by epoch
            runs_in_env = group.groupby('run_label')
            seqs = [run_group[['epoch', 'step_size', 'seed']].set_index('epoch').sort_index() for name, run_group in runs_in_env]

            # All seeds have the same X
            x = np.array([seq.index.to_numpy() for seq in seqs])[0, :]

            # Get mean and variance across seeds
            seqs_numpy = np.array([seq['step_size'].to_numpy() for seq in seqs])
            seqs_std = seqs_numpy.std(0)
            y = seqs_numpy.mean(0)

            # Render plot
            plt.title(f'Showing step-size over iterations for {env_name}')
            plt.plot(x, y, label=name, **graph_style)
            plt.fill_between(x[0: -1: 99], (y + seqs_std)[0: -1: 99], (y-seqs_std)[0: -1: 99], alpha=0.5, facecolor=graph_style['color'])
            plt.xlabel('Iterations', fontsize = 15, weight = 'bold')
            plt.ylabel('Step-size', fontsize = 15, weight = 'bold')

        plt.legend(loc='best')
        plt.savefig(f'{path_to_plots}/step_size-{env_name}.png')

if __name__ == "__main__":
    # Prep right directories
    utils.ensure_path(path_to_total_results)
    utils.ensure_path(f'{path_to_plots}/step-size.png')
    
    if merge_single_runs:
        all_runs = merge_all_single_runs(filter_unfinished=True)
    else:
        try:
            all_runs = pd.read_csv(path_to_total_results)
        except Exception as e:
            print('Path to total results do not exist. Will run `merge_all_single_runs`')
            all_runs = merge_all_single_runs(filter_unfinished=True)

    plot_stepsize_per_env(all_runs, ['Swimmer-v2'])
