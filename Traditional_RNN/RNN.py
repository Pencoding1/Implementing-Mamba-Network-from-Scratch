from typing import Tuple
from torch import Tensor, zeros
from torch.nn import Module
from Architecture_cell.RNN_Cell import RNN_Cell

class RNN(Module):
    def __init__(self, 
                 input_dim:int, 
                 hidde_dim:int, 
                 device:str="cpu",
                 bias:bool=True) -> None:
        super().__init__()
        self.cell = RNN_Cell(input_dim, hidde_dim, bias)
        self.device = device
        self.hidden_dim = hidde_dim
        
    def forward(self, sequences:Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_lengths, _ = sequences.shape
        hidden_state = zeros(batch_size, 
                             self.hidden_dim,
                             device=self.device,
                             dtype=sequences.dtype)
        
        all_hidden_state = []
        for t in range(seq_lengths):
            x_t = sequences[:, t, :]
            hidden_state = self.cell.forward(x_t, hidden_state)
            all_hidden_state.append(hidden_state)
        
        return all_hidden_state, hidden_state