from dataclasses import dataclass
import gym

@dataclass
class Config:
    env: gym.Env