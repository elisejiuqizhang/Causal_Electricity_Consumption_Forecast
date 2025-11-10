import torch
import torch.nn as nn
    
    
class GRUForecaster(nn.Module):
    """
    Returns (B, C, H) from input (B, C_in, L) with batch_first=True.
    This matches PyTorch's documented GRU I/O when using batch_first. 
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, horizon=24):
        super().__init__()
        self.horizon = horizon
        self.out_dim = output_size
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                        
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * output_size)
        )

    def forward(self, x):                            # x: (B, C_in, L)
        x = x.transpose(1, 2)                        # --> (B, L, C_in)
        y, _ = self.rnn(x)                           # y: (B, L, hidden)
        h_last = y[:, -1, :]                         # (B, hidden)
        out = self.head(h_last)                      # (B, H*C)
        out = out.view(x.size(0), self.out_dim, self.horizon)  # (B, C, H)
        return out



class LSTMForecaster(nn.Module):
    """
    Returns (B, C, H) from input (B, C_in, L) with batch_first=True.
    This matches PyTorch's documented LSTM I/O when using batch_first. 
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, horizon=24):
        super().__init__()
        self.horizon = horizon
        self.out_dim = output_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                        
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * output_size)
        )

    def forward(self, x):                            # x: (B, C_in, L)
        x = x.transpose(1, 2)                        # --> (B, L, C_in)
        y, _ = self.rnn(x)                           # y: (B, L, hidden)
        h_last = y[:, -1, :]                         # (B, hidden)
        out = self.head(h_last)                      # (B, H*C)
        out = out.view(x.size(0), self.out_dim, self.horizon)  # (B, C, H)
        return out


    