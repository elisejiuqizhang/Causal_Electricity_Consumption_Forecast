import math
import torch
import torch.nn as nn

class PatchTSTModel(nn.Module):
    """
    Patch Time Series Transformer (PatchTST) Model.
    Inputs: x -> (batch, C_in, L)
    Outputs: y -> (batch, num_targets, H)
    """
    def __init__(self, 
                 input_channels, 
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
        self.input_channels = input_channels
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
            nn.Linear(patch_len, d_model) for _ in range(input_channels)
        ])
        
        # Positional encoding for patch positions (fixed sinusoidal encoding)
        max_seq_len = max(patch_num, 512)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)  # not trainable, for adding to patches
        
        # Transformer encoder: stacks of self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                  dim_feedforward=d_ff, dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction head: linear mapping from flattened encoder outputs to forecast horizon
        # Input features to linear = C_in * patch_num * d_model, output = num_targets * H
        self.head_linear = nn.Linear(input_channels * patch_num * d_model, output_horizon * num_targets)
    
    def forward(self, x):
        """
        x: Tensor of shape (B, C_in, L)
        Returns: Tensor of shape (B, C_out, H)
        """
        B, C, L = x.shape
        if L != self.context_length:
            # The model was configured for a specific context_length
            raise RuntimeError(f"Input length {L} != model.context_length {self.context_length}")
        
        # Use positional encoding for `patch_num` length on the device of x
        pe = self.positional_encoding[:self.patch_num, :].to(x.device)  # shape (patch_num, d_model)
        
        # 1. Patch Extraction: split each channel's sequence into patches
        # Flatten batch and channel dims to apply unfold in one go
        x_flat = x.view(B * C, L)  # shape (B*C, L)
        # Determine if padding is needed for the last patch
        total_length_needed = (self.patch_num - 1) * self.patch_stride + self.patch_len
        if total_length_needed > L:
            # Pad with last value to reach the needed length
            pad_length = total_length_needed - L
            last_val = x_flat[:, -1:].detach()
            pad_tensor = last_val.repeat(1, pad_length)  # repeat last value
            x_flat = torch.cat([x_flat, pad_tensor], dim=1)  # now length = total_length_needed
        # Unfold into patches: shape (B*C, patch_num, patch_len)
        patches = x_flat.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches[:, :self.patch_num, :]  # ensure correct number of patches
        # Reshape to (B, C, patch_num, patch_len)
        patches = patches.view(B, C, self.patch_num, self.patch_len)
        
        # 2. Patch Embedding: apply channel-specific linear to each patch
        # x_embed shape: (B, C, patch_num, d_model)
        embed_list = []
        for ch in range(C):
            # Linear mapping for channel `ch`: (B, patch_num, patch_len) -> (B, patch_num, d_model)
            emb = self.patch_embeds[ch](patches[:, ch, :, :])  # apply along last dim
            embed_list.append(emb)
        x_embed = torch.stack(embed_list, dim=1)  # stack embeddings for all channels
        
        # 3. Add positional encoding to patch embeddings
        # Broadcast `pe` (patch_num, d_model) to (B, C, patch_num, d_model) and add
        x_embed = x_embed + pe.unsqueeze(0).unsqueeze(0)  # add positional encoding
        
        # 4. Transformer Encoder: process each channel's patch sequence independently
        # Prepare input for encoder: flatten channels into batch dimension
        x_enc_in = x_embed.view(B * C, self.patch_num, self.d_model)  # shape (B*C, patch_num, d_model)
        # Transpose to shape (patch_num, B*C, d_model) as expected by PyTorch's Transformer
        x_enc_in = x_enc_in.permute(1, 0, 2)  # (S, N, E) where S=patch_num, N=B*C
        # Forward through Transformer encoder (self-attention over patches, per channel sequence)
        x_enc_out = self.encoder(x_enc_in)  # shape (patch_num, B*C, d_model)
        # Transpose back and reshape: (B*C, patch_num, d_model) -> (B, C, patch_num, d_model)
        x_enc_out = x_enc_out.permute(1, 0, 2).contiguous().view(B, C, self.patch_num, self.d_model)
        
        # 5. Prediction: flatten encoder outputs and apply linear head to get forecasts
        # Flatten all channels and patches: shape (B, C * patch_num * d_model)
        x_flattened = x_enc_out.reshape(B, C * self.patch_num * self.d_model)
        pred = self.head_linear(x_flattened)  # shape (B, num_targets * H)
        # Reshape to (B, num_targets, H)/(B,C,H)
        if self.num_targets > 1:
            pred = pred.view(B, self.num_targets, self.output_horizon)
        else:
            pred = pred.view(B, 1, self.output_horizon)
        return pred
