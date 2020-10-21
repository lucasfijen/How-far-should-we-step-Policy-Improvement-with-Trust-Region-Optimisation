from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    env: str = 'Hopper-V2'
    seed: Optional[int] = None
    gamma: float = 0.995