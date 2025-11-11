from torch.nn import Module, Parameter
from torch import Tensor

class Sequence_Model(Module):
    def __init__(self, input_dim: int, hidden_dim:int, device:str) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device=device
    
    def forward(self, x:Tensor) -> Tensor:...
