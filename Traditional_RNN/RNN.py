from typing import Tuple
from torch import Tensor, zeros, stack
from torch.nn import Module
from Architecture_cell.RNN_Cell import RNN_Cell

class RNN(Module):
    def __init__(self, 
                 input_dim:int, 
                 hidden_dim:int, 
                 device:str="cpu",
                 bias:bool=True) -> None:
        super().__init__()
        self.cell = RNN_Cell(input_dim, hidden_dim, bias)
        self.device = device
        self.hidden_dim = hidden_dim
        
    def forward(self, sequences:Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_lengths, _ = sequences.shape
        hidden_state = zeros(batch_size, 
                             self.hidden_dim,
                             device=self.device,
                             dtype=sequences.dtype)
        
        all_hidden_states = []
        for t in range(seq_lengths):
            x_t = sequences[:, t, :]
            hidden_state = self.cell(x_t, hidden_state)
            all_hidden_states.append(hidden_state)        

        all_hidden_states_tensor = stack(all_hidden_states, dim=1)
        
        return all_hidden_states_tensor, hidden_state