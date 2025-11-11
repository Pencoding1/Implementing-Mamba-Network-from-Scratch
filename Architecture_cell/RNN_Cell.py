from Sequence_Model import Sequence_Model
from torch.nn import Parameter, init
from torch import Tensor, math, empty, zeros, tanh

class RNN_Cell(Sequence_Model):
    def __init__(self, 
                 input_dim:int, 
                 hidden_dim:int, 
                 bias:bool = True,
                 device:str='cpu'
                ) -> None:
        """
            _Args:
            _Description:
            _Returns:
        """
        super().__init__(input_dim, hidden_dim, device)
        self.U = Parameter(empty(input_dim, hidden_dim))
        self.W = Parameter(empty(hidden_dim, hidden_dim))
        
        if bias:
            bias_x_tensor = empty(hidden_dim)
            bias_h_tensor = empty(hidden_dim)
            self.bias_x = Parameter(bias_x_tensor)
            self.bias_h = Parameter(bias_h_tensor)
        else:
            self.register_parameter('bias_x', None)
            self.register_parameter('bias_h', None)
    
    def _reset_parameters(self) -> None:
        """
            Initializes weights with a Uniform distribution
        """
        k = 1.0 / self.hidden_dim
        bound = math.sqrt(k)
        init.uniform_(self.W, -bound, bound)
        init.uniform_(self.U, -bound, bound)
        
        if self.bias_x is not None:
            init.uniform_(self.bias_x, -bound, bound)
            init.uniform_(self.bias_h, -bound, bound)
    
    def forward(self, X: Tensor, hidden_state: Tensor=None) -> Tensor:
        if X.device != self.device:
            X = X.to(self.device)

        if hidden_state is None:
            hidden_state = zeros(X.size(0), 
                                 self.hidden_dim, 
                                 device=self.device,
                                 dtype=X.dtype
                                 )
        else:
            hidden_state = hidden_state.to(self.device)

        compressed_input = X @ self.U # (batch x embedding_size) @ (embedding_size x hidden_size) = (batch x hidden_size) 
        if self.bias_x is not None:
            compressed_input = compressed_input + self.bias_x

        compressed_state = hidden_state @ self.W
        if self.bias_h is not None:
            compressed_state = compressed_state + self.bias_h   

        new_hidden_state = tanh(compressed_input + compressed_state)
        return new_hidden_state
