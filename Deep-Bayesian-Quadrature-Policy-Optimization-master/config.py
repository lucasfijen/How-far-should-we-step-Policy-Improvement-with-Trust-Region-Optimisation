import torch
from enum import Enum
from typing import Optional, NamedTuple, List

from arguments import get_args

args, _ = get_args()

def args_or_default(args_key, default=None):
    args_dict = args.__dict__

    return args_dict[args_key] if args_key in args_dict else default


class Config(NamedTuple):
    label: str = args_or_default('label', '')

    pg_models: List[str] = [args_or_default('pg_algorithm', 'TRPO')]
    env: str = args_or_default('env-name', 'Hopper-v2')
    seed: Optional[int] = args_or_default('seed', -1)
    gamma: float = args_or_default('gamma', -1)

    device: str = 'cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu'
    path_to_results: str = args_or_default('output_directory', 'results')

    npg_lr: float = args_or_default('lr', 7e-4)