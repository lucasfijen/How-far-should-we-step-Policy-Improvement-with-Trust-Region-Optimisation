from genericpath import exists
from PIL import Image
import imageio
import utils
import os
import csv
import pandas as pd
from tensorboardX import SummaryWriter
from typing import NamedTuple

class ResultsWriterOptions(NamedTuple):
    pg_model: str

class ResultsRow(NamedTuple):
    env: str
    seed: int
    run_label: str
    run_nr_epochs: str
    nr_episodes: str
    run_model: str
    timestamp: str
    epoch: int
    step_size: float
    epoch_duration: float
    nr_steps: int
    perf: float
    hardware: str 

class ResultsWriter:
    def __init__(self, run_label: str, path_to_results: str, pg_model: str = '') -> None:
        self.path = path_to_results
        self.run_label: str = run_label
        utils.ensure_path(f'{path_to_results}/results.csv')

        utils.ensure_path(f'{path_to_results}/{run_label}/results.csv')

        self.tensorboard_writer = SummaryWriter(
            f'{path_to_results}/tboard_logs',
            comment=f'pg_{pg_model}'
        )

        self.columns = [
            'env',
            'seed',
            'run_label',
            'run_nr_epochs',
            'nr_episodes',
            'run_model', 
            'timestamp', 
            'epoch',
            'step_size',
            'epoch_duration',
            'nr_steps', 
            'perf',
            'hardware'
        ]

        self.results = pd.DataFrame(columns=self.columns)

        if (not os.path.exists(f'{path_to_results}/results.csv') or os.path.getsize(f'{path_to_results}/results.csv') == 0):
            with open(f'{self.path}/results.csv','w') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def add(self, results: ResultsRow):
        """Store results in a CSV"""
        self.results = self.results.append(results._asdict(), ignore_index=True)
        
        # Write to shared csv
        with open(f'{self.path}/results.csv','a') as f:
            writer = csv.writer(f)
            writer.writerow(results)

        # Also store in run-specific csv
        self.results.to_csv(f'{self.path}/{self.run_label}/results.csv', mode='w')

        # Store 
        self.tensorboard_writer.add_scalar("Average reward", results.perf, results.step_size)
    
    def save_render(self, renders_array, iteration):
        """Save GIF of an array of np arrays representing images"""
        render_images = [Image.fromarray(i) for i in renders_array]
        imageio.mimsave(f'{self.path}/{self.run_label}/simulation-iter-{iteration}.gif', render_images, 'GIF')