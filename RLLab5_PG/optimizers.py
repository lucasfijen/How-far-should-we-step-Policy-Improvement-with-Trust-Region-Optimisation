import torch
import torch.optim as optim

class TRPO(optim.Optimizer):
    def __init__(self, params: torch.Tensor) -> None:
        self.params = params

    @torch.no_grad()
    def step(self):
        