import torch
import torch.nn as nn
try:
    # New recommended API
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    # Fallback for older torch
    from torch.nn.utils import weight_norm


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        horizon,
        num_targets=1,          
        hidden_channels=64,
        levels=4,
        kernel_size=3,
        dropout=0.2,
        dilation_base=2,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.output_horizon  = horizon
        self.num_targets     = num_targets

        layers = []
        in_ch = input_dim             # Conv1d in_channels == feature dim D
        dilation = 1
        for _ in range(levels):
            out_ch = hidden_channels
            pad = (kernel_size - 1) * dilation  # causal padding, will trim after conv
            conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                             dilation=dilation, padding=pad)
            # Use parametrizations.weight_norm (new) if available
            conv = weight_norm(conv)
            layers += [conv, nn.ReLU(), nn.Dropout(dropout)]
            in_ch = out_ch
            dilation *= dilation_base

        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(hidden_channels, self.output_horizon * num_targets)

    def forward(self, x):
        """
        x: (B, L, D) from the dataloader
        returns: (B, horizon, num_targets)
        """
        # Conv1d expects (B, C, L) => transpose (B, L, D) -> (B, D, L)
        if x.dim() != 3:
            raise RuntimeError(f"Expected 3D input (B,L,D), got {tuple(x.shape)}")
        x = x.transpose(1, 2)  # (B, D, L)
        L = x.size(-1)

        y = self.tcn(x)       # (B, hidden, L + extra)
        # Causal padding adds future time; trim back to original length
        if y.size(-1) != L:
            y = y[..., :L]    # keep only past+current

        # Use the last time step features
        last_step = y[:, :, -1]           # (B, hidden)
        out = self.fc(last_step)          # (B, horizon * num_targets)
        out = out.view(out.size(0), self.output_horizon, self.num_targets)
        return out
