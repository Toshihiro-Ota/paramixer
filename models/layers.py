"""
Copyright (c) 2024. Toshihiro Ota
Licensed under the Apache License, Version 2.0 (the "License");

MLP modules w/ serial (vanilla) & parallelized MixerBlocks

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple


class vMlp(nn.Module):
    """ vanilla MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=False,  #* set True for vanillamixer
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# symmetric MLP
class sMlp(nn.Module):
    """ MLP with symmetric weights from the Hopfield/Mixer correspondence
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=False,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc2 = linear_layer(in_features, hidden_features, bias=bias[0])  # rename as fc2 for compatibility with other models
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])

    # symmetric feedforward MLP
    def forward(self, x):
        x = F.linear(x, self.fc2.weight)
        x = self.act(x)
        x = self.drop1(x)
        x = F.linear(x, self.fc2.weight.T)
        x = self.drop2(x)
        return x


# asymmetric MLP
class aMlp(nn.Module):
    """ MLP with asymmetric weights for symmetry breaking phase
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=False,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    # asymmetric feedforward MLP
    def forward(self, x):
        x = F.linear(x, self.fc1.weight)
        x = self.act(x)
        x = self.drop1(x)
        x = F.linear(x, self.fc1.weight.T) + self.fc2(x)  #* symmetry breaking term
        x = self.drop2(x)
        return x


# vanilla (serial) MixerBlock w/ iterative forward operation
class vMixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=vMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
            hidden_ratio=1,
            n_iter=1,
    ):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer((seq_len, dim))  #! symmetric layernorm
        self.mlp_tokens = mlp_layer(seq_len, int(tokens_dim*hidden_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((seq_len, dim))  #! symmetric layernorm
        self.mlp_channels = mlp_layer(dim, int(channels_dim*hidden_ratio), act_layer=act_layer, drop=drop)

        self.n_iter = n_iter

    def forward(self, x):
        for _ in range(self.n_iter):
            x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
            x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


# parallelized MixerBlock w/ iterative forward operation
class pMixerBlock(nn.Module):
    """ Parallelized MixerBlock from the Hopfield/Mixer correspondence
    """
    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=vMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
            hidden_ratio=1,
            n_iter=1,
    ):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm = norm_layer((seq_len, dim))  #! symmetric layernorm
        self.mlp_tokens = mlp_layer(seq_len, int(tokens_dim*hidden_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_channels = mlp_layer(dim, int(channels_dim*hidden_ratio), act_layer=act_layer, drop=drop)

        self.n_iter = n_iter

    # parallelized feedforward MLPs
    def forward(self, x):
        for _ in range(self.n_iter):
            x = x + self.drop_path(self.mlp_tokens(self.norm(x).transpose(1, 2)).transpose(1, 2)) + self.drop_path(self.mlp_channels(self.norm(x)))
        return x
