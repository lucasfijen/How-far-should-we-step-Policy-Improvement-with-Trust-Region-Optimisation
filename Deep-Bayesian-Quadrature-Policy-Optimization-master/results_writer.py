import utils
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
    nr_steps: int
    velocity: float
    perf: float    

class ResultsWriter:
    def __init__(self, path_to_results: str, pg_model: str = '') -> None:
        self.path = path_to_results
        utils.ensure_path(path_to_results)

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
            'step_size', 
            'nr_steps', 
            'velocity', 
            'perf'
        ]
        self.results = pd.DataFrame(columns=self.columns)

    def add(self, results: ResultsRow):
        self.results = self.results.append(results)
        self.results.to_csv(self.path)

        self.tensorboard_writer.add_scalar("Average reward", results.perf, results.step_size)
        
