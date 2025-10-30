import torch
import torch.nn as nn
import torch.nn.functional as F
import mambapy
from mambapy.mamba import Mamba, MambaConfig

# multivariate forecaster
class ForecastHead(nn.Module):
    def __init__(self, d_model, horizon, num_cities):
        super().__init__()
        self.proj = nn.Linear(d_model, horizon * num_cities)
        self.horizon, self.num_cities = horizon, num_cities
    def forward(self, h_last):  # (B, d_model)
        # y = self.proj(h_last)
        # return y.view(-1, self.horizon, self.num_cities)
        y = self.proj(h_last).view(-1, self.horizon, self.num_cities)  # (B, H, C)
        return y.permute(0, 2, 1)  # (B, C, H)
    
class MambaForecaster(nn.Module):
    def __init__(
        self, 
        in_dim, 
        d_model, 
        n_layers, 
        d_state, 
        horizon, 
        mode='multi', 
        num_cities=None, 
        d_conv=4, 
        expand=2
    ):
        """
        mode: 'multi' for multivariate forecasting, 'uni' for univariate forecasting
        num_cities: required if mode == 'multi'
        """
        super().__init__()
        cfg = MambaConfig(
            d_model=d_model, 
            n_layers=n_layers, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand_factor=expand
        )
        self.embed = nn.Linear(in_dim, d_model)
        self.backbone = Mamba(cfg)
        self.mode = mode
        self.horizon = horizon
        if mode == 'multi':
            assert num_cities is not None, "num_cities must be specified for multivariate forecasting"
            self.num_cities = num_cities
            self.head = ForecastHead(d_model, horizon, num_cities)
        elif mode == 'uni':
            self.num_cities = 1
            self.head = nn.Linear(d_model, horizon)
        else:
            raise ValueError("mode must be either 'multi' or 'uni'")

    def forward(self, x):
        # x: (B, C_in, L) or (B, Lin, in_dim)
        x = x.transpose(1, 2)  # -> (B, L, C_in)
        x = self.embed(x)      # (B, L, d_model)
        y = self.backbone(x)   # (B, L, d_model)
        h_last = y[:, -1, :]   # (B, d_model)
        if self.mode == 'multi':
            out = self.head(h_last)  # (B, C, H)
            out = out.permute(0, 2, 1)  # (B, H, C)
        else:
            out = self.head(h_last).unsqueeze(-1)  # (B, H, 1)
        return out  # (B, H, C)
