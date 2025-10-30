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
    
class MultiVForecaster(nn.Module):
    def __init__(self, in_dim, d_model, n_layers, d_state, horizon, num_cities, d_conv=4, expand=2):
        super().__init__()
        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=d_state, d_conv=d_conv, expand_factor=expand)
        self.embed = nn.Linear(in_dim, d_model)
        self.backbone = Mamba(cfg)   # maps (B,L,d_model)->(B,L,d_model)
        self.head = ForecastHead(d_model, horizon, num_cities)

    def forward(self, x):  # x: (B, Lin, in_dim)
        # x = self.embed(x)
        # y = self.backbone(x)          # (B, Lin, d_model)
        # h_last = y[:, -1, :]          # last step
        # return self.head(h_last)      # (B, H, C)

        x = x.transpose(1, 2)             # --> (B, L, C_in)
        x = self.embed(x)                 # (B, L, d_model)
        y = self.backbone(x)              # (B, L, d_model)
        h_last = y[:, -1, :]              # (B, d_model)
        # out = self.head(h_last)           # default (B, H, C), see fix #2
        # return out.permute(0, 2, 1)       # --> (B, C, H) to match your targets

        return self.head(h_last)      # (B, H, C)



# total (univar) forecaster
class UniVForecaster(nn.Module):
    def __init__(self, in_dim, d_model, n_layers, d_state, horizon, d_conv=4, expand=2):
        super().__init__()
        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=d_state, d_conv=d_conv, expand_factor=expand)
        self.embed = nn.Linear(in_dim, d_model)
        self.backbone = Mamba(cfg)     # (B,L,d_model)->(B,L,d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):  # x: (B, C_in, L)
        x = x.transpose(1, 2)          # -> (B, L, C_in)
        x = self.embed(x)              # (B, L, d_model)
        y = self.backbone(x)           # (B, L, d_model)
        h_last = y[:, -1, :]           # (B, d_model)
        return self.head(h_last)       # (B, H)
