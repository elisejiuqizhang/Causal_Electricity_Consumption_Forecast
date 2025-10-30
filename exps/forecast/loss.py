import numpy as np
import torch
import torch.nn as nn


class PerChannelLoss(nn.Module):
    def __init__(self, loss_type='mse', reduction='mean', per_channel=False, channel_weights=None, eps=1e-8):
        super().__init__()
        assert isinstance(loss_type, str), "loss_type should be a string."
        assert isinstance(reduction, str), "reduction should be a string"
        assert loss_type.lower() in ['mse', 'mae', 'l1', 'l2', 'rmse']
        assert reduction.lower() in ['mean', 'sum', 'none']
        self.loss_type=loss_type
        self.reduction=reduction
        self.per_channel=per_channel
        self.eps=eps

        if channel_weights is not None:
            w=torch.as_tensor(channel_weights, dtype=torch.float32)
            self.register_buffer('channel_weights', w/w.sum())
        else:
            self.register_buffer('channel_weights', None)

    def forward(self, pred, tgt):
        """ Shape: (bs, chnl, len_ts)"""
        
        assert len(pred.shape)==3, "pred should have dimensions : (batch size, channels, length of time series)"
        assert len(tgt.shape)==3, "target should have dimensions : (batch_size, channels, length of time series)"
        assert pred.shape[0]==tgt.shape[0], "batch size (dim0) should match"
        assert pred.shape[1]==tgt.shape[1], "channel size (dim1) should match"
        assert pred.shape[2]==tgt.shape[2], "sequence length (dim2) should match"

        diff=pred-tgt # Shape: (bs, chnl, length)

        # mean over batch and time for each channel
        # output shape: (C,)
        if self.loss_type in ['mse', 'l2']:
            per_ch = diff.pow(2).mean(dim=(0,2))
        elif self.loss in ['mae', 'l1']:
            per_ch = diff.abs().mean(dim=(0,2))
        else: # rmse
            per_ch=torch.sqrt(diff.pow(2).mean(dim=(0,2))+self.eps)

        if self.channel_weights is not None:
            per_ch=per_ch*self.channel_weights

        if self.per_channel or self.reduction == 'None':
            return per_ch
        
        if self.reduction == 'mean':
            return per_ch.mean()
        else:
            return per_ch.sum()



        