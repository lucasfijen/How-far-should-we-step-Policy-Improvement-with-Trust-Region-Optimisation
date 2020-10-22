import utils
import csv
import pandas as pd
from tensorboardX import SummaryWriter
from typing import NamedTuple

class ResultsWriterOptions(NamedTuple):
    pg_model: str

class ResultsRow(NamedTuple):
    run_label: str
    run_nr_epochs: str
    run_nr_rollouts: str
    run_model: str
    timestamp: str
    iteration: int
    step_size: float
    iteration_duration: float
    nr_steps: int
    velocity: float
    perf: float
    hardware: str 

class ResultsWriter:
    def __init__(self, run_label: str, path_to_results: str, pg_model: str = '') -> None:
        self.path = path_to_results
        self.run_label: str = run_label
        utils.ensure_path(f'{path_to_results}/results.csv')
        utils.ensure_path(f'{path_to_results}/{run_label}-results.csv')

        self.tensorboard_writer = SummaryWriter(
            f'{path_to_results}/tboard_logs',
            comment=f'pg_{pg_model}'
        )

        self.columns = [
            'run_label',
            'run_nr_epochs',
            'run_nr_rollouts',
            'run_model', 
            'timestamp', 
            'iteration',
            'iteration_duration',
            'step_size', 
            'nr_steps', 
            'velocity', 
            'perf',
            'hardware'
        ]

        self.results = pd.DataFrame(columns=self.columns)

    def add(self, results: ResultsRow):
        self.results = self.results.append(results._asdict(), ignore_index=True)
        
        # Write to shared csv
        with open(f'{self.path}/results.csv','a') as f:
            writer = csv.writer(f)
            writer.writerow(results)

        # Also store in run-specific csv
        self.results.to_csv(f'{self.path}/{self.run_label}-results.csv', mode='w')

        # Store 
        self.tensorboard_writer.add_scalar("Average reward", results.perf, results.step_size)
        
    
