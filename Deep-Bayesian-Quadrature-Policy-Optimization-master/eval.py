import pandas as pd
import matplotlib.pyplot as plt
import utils
from pathlib import Path
import seaborn as sns

plt.style.use('seaborn-dark')

path_to_results = 'temp_results'
path_to_plots = 'temp_results/plots'
path_to_total_results = 'temp_results/total.csv'
merge_single_runs = False

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
    for env_name in envs:
        env_runs_df = all_runs_df[all_runs['env'] == env_name]
        env_runs_per_run_df = env_runs_df.groupby('run_label')

        for name, group in env_runs_per_run_df:
            color = 'blue'
            linestyle = '- - -'
            alpha = 1

            if group.iloc[0]['run_model'] == 'NPG':
                linestyle = 'solid'
                alpha = 0.3
            elif group.iloc[0]['run_model'] == 'TRPO':
                linestyle = 'dashdot'
            else:
                linestyle = 'dashdot'
            
            if group.iloc[0]['seed'] == 42:
                # Blueish
                color = '#7F9CF5'
            elif group.iloc[0]['seed'] == 666:
                # Redish
                color = '#FC8181'
            else: # seed is 1337
                # Greenish
                color = '#68D391'

            linestyle = '--' if group.iloc[0]['run_model'] == 'NPG' else '-'
            plt.plot(group['epoch'], group['step_size'], linestyle=linestyle, color=color, alpha=alpha, label=name)

        plt.legend(loc='best')
        plt.savefig(f'{path_to_plots}/step_size-{env_name}.png')
        print('done')
        
        # sns_plot = sns.lineplot(data=env_runs_df, x='epoch', y='step_size', style='run_label')
        # sns_plot.savefig(f'{path_to_plots}/step_size-{env_name}.png')
        

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
