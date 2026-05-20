import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.mlp import MLP
from models.nir import NIR


def get_learnable_transforms(num_samples, coordinate_dim=2, zeros=True, freeze_first=True):
    if zeros:
        if freeze_first:
            params = [nn.Parameter(torch.zeros(1, coordinate_dim), requires_grad=(i != 0)) for i in range(num_samples)]
        else:
            params = [nn.Parameter(torch.zeros(1, coordinate_dim), requires_grad=True) for i in range(num_samples)]
    else:
        if freeze_first:
            params = [nn.Parameter(torch.ones(1, coordinate_dim), requires_grad=(i != 0)) for i in range(num_samples)]
        else:
            params = [nn.Parameter(torch.ones(1, coordinate_dim), requires_grad=True) for i in range(num_samples)]
    return nn.ParameterList(params)


def get_learnable_affines(num_samples, freeze_first=True):
    """Return per-sample learnable 2x3 affine params initialized to identity."""
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    params = []
    for i in range(num_samples):
        params.append(nn.Parameter(identity.clone(), requires_grad=(i != 0) if freeze_first else True))
    return nn.ParameterList(params)


def get_direct_variances(num_samples, dimensions, freeze_first=True):
    if freeze_first:
        params = [nn.Parameter(torch.zeros(dimensions) if i != 0 else nn.Parameter(torch.zeros(dimensions)), requires_grad=(i != 0)) for i in range(num_samples)]
    else:
        params = [nn.Parameter(torch.zeros(dimensions) if i != 0 else nn.Parameter(torch.zeros(dimensions)), requires_grad=True) for i in range(num_samples)]
    return nn.ParameterList(params)


def get_decoder(
    network_name,
    network_depth,
    input_dim,
    network_hidden_dim,
    output_dim=3,
    tcnn_mlp_dtype="fp16",
    mlp_init="kaiming",
    device=None,
    hash_n_levels=None,
    hash_n_features_per_level=None,
    hash_attn_token_dim=32,
    attn_token_dim=32,
    fourier_num_bands=8,
):
    if network_name == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dim=network_hidden_dim,
            depth=network_depth,
            output_dim=output_dim,
            init_scheme=mlp_init,
        )
    elif network_name == "mlp_tcnn":
        from models.mlp_tcnn import MLPTcnn

        return MLPTcnn(
            input_dim=input_dim,
            hidden_dim=network_hidden_dim,
            depth=network_depth,
            output_dim=output_dim,
            dtype=tcnn_mlp_dtype,
            device=device,
        )
    elif network_name == "nir":
        return NIR(input_dim=input_dim, hidden_dim=network_hidden_dim, depth=network_depth, output_dim=output_dim)
    elif network_name == "hash_attn":
        from models.hash_attention_decoder import HashLevelAttentionDecoderLite

        if hash_n_levels is None or hash_n_features_per_level is None:
            raise ValueError("hash_attn requires hash_n_levels and hash_n_features_per_level")
        return HashLevelAttentionDecoderLite(
            n_levels=hash_n_levels,
            n_features_per_level=hash_n_features_per_level,
            token_dim=hash_attn_token_dim,
            hidden_dim=network_hidden_dim,
            out_dim=output_dim,
        )
    elif network_name == "fourier_band_attn":
        from models.fourier_band_attention_decoder import FourierBandAttentionDecoderLite

        return FourierBandAttentionDecoderLite(
            input_dim=input_dim,
            num_bands=fourier_num_bands,
            token_dim=attn_token_dim,
            hidden_dim=network_hidden_dim,
            out_dim=output_dim,
        )
    else:
        raise ValueError(
            f"Network name {network_name} not recognized. "
            "Use 'mlp', 'mlp_tcnn', 'nir', 'hash_attn', or 'fourier_band_attn'."
        )
