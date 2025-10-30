import torch
import torch.nn as nn

class TCNModel(nn.Module):
    """
    Inputs:  x  -> (batch, C_in, L)
    Outputs: y  -> (batch, num_targets, H)
    """
    def __init__(
        self,
        input_channels,
        output_horizon,
        num_targets=4,
        hidden_channels=64,
        levels=4,
        kernel_size=3,
        dropout=0.2,
        dilation_base=2,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.output_horizon  = output_horizon
        self.num_targets     = num_targets

        layers = []
        in_ch = input_channels
        dilation = 1
        for _ in range(levels):
            out_ch = hidden_channels
            pad = (kernel_size - 1) * dilation
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
            layers += [nn.utils.weight_norm(conv), nn.ReLU(), nn.Dropout(dropout)]
            in_ch = out_ch
            dilation *= dilation_base

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels, output_horizon * num_targets)
        self.num_targets = num_targets
        self.output_horizon = output_horizon

    def forward(self, x):
        # x: (B, C, L)
        y = self.tcn(x)               # (B, hidden, L')

        if y.dim()==3:
            # last_step = y[:, :, -1]       # (B, hidden)

            B, C, T = y.shape
            if C == self.hidden_channels:
                last_step = y[:, :, -1]      # (B, hidden)
            elif T == self.hidden_channels:
                # Got (B, L', hidden): time/channel swapped by upstream block
                last_step = y[:, -1, :]      # (B, hidden)
            else:
                raise RuntimeError(f"Unexpected 3D shape from tcn: {tuple(y.shape)}; "
                                   f"hidden={self.hidden_channels}")
            
        elif y.dim()==2:
            # last_step=y

            # Could be (B, hidden) OR (hidden, L')
            B, F = y.shape
            if F == self.hidden_channels:
                last_step = y                 # (B, hidden)
            elif B == self.hidden_channels:
                # Treat as (hidden, L'): take last time and add batch dim = 1
                last_step = y[:, -1].unsqueeze(0)  # (1, hidden)
            else:
                raise RuntimeError(f"Unexpected 2D shape from tcn: {tuple(y.shape)}; "
                                   f"hidden={self.hidden_channels}")

        else:
            raise RuntimeError(f"Unexpected shape from tcn: {tuple(y.shape)}")
        out = self.fc(last_step)      # (B, num_targets*H)
        out = out.view(out.size(0), self.num_targets, self.output_horizon)
        return out