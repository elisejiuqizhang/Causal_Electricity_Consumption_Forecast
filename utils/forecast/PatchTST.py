import math
import torch
import torch.nn as nn

class PatchTSTModel(nn.Module):
    """
    Patch Time Series Transformer (PatchTST) Model.
    Inputs (from loader): x -> (B, L, D)   # <- changed: accept (B, L, D)
    Internally: we transpose to (B, C_in, L) for patching per channel.
    Outputs: y -> (B, H, num_targets)  [your trainer will squeeze if last dim==1]
    """
    def __init__(self, 
                 input_dim, 
                 output_horizon, 
                 num_targets=1, 
                 context_length=168, 
                 d_model=32, 
                 n_heads=4, 
                 n_layers=3, 
                 patch_len=16, 
                 patch_stride=8, 
                 d_ff=128, 
                 dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_horizon = output_horizon
        self.num_targets = num_targets
        self.context_length = context_length
        self.d_model = d_model
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        
        # Calculate number of patches for a sequence of length `context_length`
        if context_length < patch_len:
            raise ValueError("context_length must be >= patch_len")
        patch_num = ((context_length - patch_len) // patch_stride) + 1
        if (context_length - patch_len) % patch_stride != 0:
            patch_num += 1  # one partial patch (will be padded)
        self.patch_num = patch_num
        
        # Channel-independent patch embedding layers (one Linear per input channel)
        self.patch_embeds = nn.ModuleList([
            nn.Linear(patch_len, d_model) for _ in range(input_dim)
        ])
        
        # Positional encoding for patch positions (fixed sinusoidal encoding)
        max_seq_len = max(patch_num, 512)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)  # not trainable
        
        # Transformer encoder (batch_first=True so we can stay (B, S, E))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True   # <- important
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction head
        self.head_linear = nn.Linear(input_dim * patch_num * d_model, output_horizon * num_targets)
    
    def forward(self, x):
        """
        x: Tensor of shape (B, L, D) from the dataloader
        Returns: Tensor of shape (B, H, num_targets)
        """
        if x.dim() != 3:
            raise RuntimeError(f"Expected (B,L,D), got {tuple(x.shape)}")

        B, L, D = x.shape
        if L != self.context_length:
            raise RuntimeError(f"Input length {L} != model.context_length {self.context_length}")
        if D != self.input_dim:
            raise RuntimeError(f"Input dim {D} != model.input_dim {self.input_dim}")

        # transpose to (B, C, L) for channel-wise patching
        x = x.transpose(1, 2)  # (B, D, L) == (B, C, L)

        # positional encoding for device
        pe = self.positional_encoding[:self.patch_num, :].to(x.device)  # (patch_num, d_model)
        
        # 1) Patch extraction per channel
        B_, C, L_ = x.shape
        x_flat = x.reshape(B_ * C, L_)  # (B*C, L)
        total_length_needed = (self.patch_num - 1) * self.patch_stride + self.patch_len
        if total_length_needed > L_:
            pad_length = total_length_needed - L_
            last_val = x_flat[:, -1:].detach()
            pad_tensor = last_val.repeat(1, pad_length)
            x_flat = torch.cat([x_flat, pad_tensor], dim=1)
        patches = x_flat.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)  # (B*C, patch_num, patch_len)
        patches = patches[:, :self.patch_num, :]
        patches = patches.view(B_, C, self.patch_num, self.patch_len)  # (B, C, patch_num, patch_len)
        
        # 2) Patch embedding (channel-specific linear)
        embed_list = []
        for ch in range(C):
            emb = self.patch_embeds[ch](patches[:, ch, :, :])  # (B, patch_num, d_model)
            embed_list.append(emb)
        x_embed = torch.stack(embed_list, dim=1)  # (B, C, patch_num, d_model)
        
        # 3) Add positional encoding
        x_embed = x_embed + pe.unsqueeze(0).unsqueeze(0)  # (B, C, patch_num, d_model)
        
        # 4) Transformer Encoder: process each channel independently
        # Flatten channels into batch dim, keep batch_first=True => (B*C, patch_num, d_model)
        x_enc_in = x_embed.view(B_ * C, self.patch_num, self.d_model)
        x_enc_out = self.encoder(x_enc_in)  # (B*C, patch_num, d_model)
        x_enc_out = x_enc_out.view(B_, C, self.patch_num, self.d_model)
        
        # 5) Prediction head
        x_flattened = x_enc_out.reshape(B_, C * self.patch_num * self.d_model)
        pred = self.head_linear(x_flattened)  # (B, H * num_targets)

        # Return (B, H, num_targets)
        pred = pred.view(B_, self.output_horizon, self.num_targets)
        return pred
