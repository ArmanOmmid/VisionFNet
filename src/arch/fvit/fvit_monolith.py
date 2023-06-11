# Base Vision Transformer Architecture Credit to PyTorch
# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.ops import MLP, Conv2dNormActivation
        
class AttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(hidden_dim, [mlp_dim, hidden_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        
        x, _ = self.self_attention(x, x, x, need_weights=True)
            
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class SpectralOperation(nn.Module):
    def __init__(self, H, W, hidden_dim):
        super().__init__()
        self.H = H
        self.W = W
        self.F = W // 2 + 1
        self.G = H * self.F
        self.hidden_dim = hidden_dim
    def forward(self, x):
        raise NotImplementedError()

class F_Linear(SpectralOperation):
    def __init__(self, H, W, hidden_dim):
        super().__init__(H, W, hidden_dim)
        self.weights = nn.Parameter(torch.empty(hidden_dim, hidden_dim, 2).normal_(std=0.02))
    def forward(self, x):
        x = torch.matmul(x, torch.view_as_complex(self.weights))
        return x

class GFN(SpectralOperation):
    def __init__(self, H, W, hidden_dim):
        super().__init__(H, W, hidden_dim)
        self.weights = nn.Parameter(torch.empty(H, self.F, hidden_dim, 2, dtype=torch.float32).normal_(std=0.02))
    def forward(self, x):
        x = x * torch.view_as_complex(self.weights)
        return x

class FNO(SpectralOperation):
    def __init__(self, H, W, hidden_dim):
        super().__init__(H, W, hidden_dim)
        self.weights = nn.Parameter(torch.empty(H, self.F, hidden_dim, hidden_dim, 2, dtype=torch.float32).normal_(std=0.02))
    def forward(self, x):
        x = torch.einsum("nhfd,hfds->nhfd", x, torch.view_as_complex(self.weights))
        return x

class F_Attention(SpectralOperation):
    def __init__(self, H, W, hidden_dim, num_heads, dropout):
        super().__init__(H, W, hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim*2, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x):
        N = x.shape[0]
        x = torch.view_as_real(x)
        x = x.reshape(N, self.H, self.F, self.hidden_dim*2)
        x = x.view(N, self.G, self.hidden_dim*2)
        x, _= self.self_attention(x, x, x)
        x = x.reshape(N, self.G, self.hidden_dim*2).reshape(N, self.H, self.F, self.hidden_dim, 2)
        x = torch.view_as_complex(x)
        return x
    
class SpectralBlock(nn.Module):
    def __init__(
        self,
        layer_encoding: int,
        sequence_lengths: List[int],
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.sequence_lengths = sequence_lengths
        self.num_heads = num_heads

        # FFT block
        self.ln_1 = norm_layer(hidden_dim)

        """
        0 = Attention (Not Included Here)
        -1 = F_Linear (Shared)
        1 = F_Linear 
        2 = GFT
        3 = FNO
        4 = Fourier Attention
        """

        self.spectral_operations = [None for _ in self.sequence_lengths]
        self.spectral_indices = [0 for _ in self.sequence_lengths]

        if layer_encoding == -1:
            self.spectral_operations[0] = F_Linear(None, None, hidden_dim)
        else:
            for i, sequence_length in enumerate(self.sequence_lengths):
                self.spectral_indices[i] = i
                H = W = int(math.sqrt(sequence_length))
                if layer_encoding == 1:
                    self.spectral_operations[i] = F_Linear(H, W, hidden_dim)
                elif layer_encoding == 2:
                    self.spectral_operations[i] = GFN(H, W, hidden_dim)
                elif layer_encoding == 3:
                    self.spectral_operations[i] = FNO(H, W, hidden_dim)
                elif layer_encoding == 4:
                    self.spectral_operations[i] = F_Attention(H, W, hidden_dim, num_heads, dropout)
                else:
                    raise NotImplementedError(f"Layer Encoding Not Mapped: {layer_encoding}")
        
        self.spectral_operations = nn.ModuleList(self.spectral_operations)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(hidden_dim, [mlp_dim, hidden_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        
        multiscale_view = list(torch.split(x, self.sequence_lengths, dim=1))
        
        for i in range(len(multiscale_view)):
            x = multiscale_view[i]
            N, L, C = x.shape
            H = W = int(math.sqrt(L))

            x = x.view(N, H, W, C)
            x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

            x = self.spectral_operations[self.spectral_indices[i]](x)

            x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
            x = x.reshape(N, L, C)
            multiscale_view[i] = x

        x = torch.cat(multiscale_view, dim=1)   
        
        x = self.dropout(x)
        # x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return y + input


class Encoder(nn.Module):
    def __init__(
        self,
        sequence_lengths: List[int],
        layer_config: list,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        seq_length = sum(sequence_lengths)
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i, layer_encoding in enumerate(layer_config):
            if layer_encoding == 0:
                layers[f"atn_layer_{i}"] = AttentionBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                )
            else:
                layers[f"spct_layer_{i}"] = SpectralBlock(
                    layer_encoding,
                    sequence_lengths,
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    norm_layer,
                )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        base_patch_size: int,
        scale_factors: List[float],
        layer_config: list,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % base_patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.base_patch_size = base_patch_size
        self.scale_factors = scale_factors
        self.layer_config = layer_config
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        patch_sizes = [int(base_patch_size * scale) for scale in scale_factors]
        self.patch_sizes = patch_sizes
        
        self.patching_filters = torch.nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
                )
                for patch_size in patch_sizes
            ]
        )

        sequence_lengths = [((image_size // p) ** 2) for p in patch_sizes]
        self.sequence_lengths = sequence_lengths

        self.encoder = Encoder(
            sequence_lengths,
            layer_config,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length = sum(self.sequence_lengths)

        reduced_tokens = int(math.sqrt(seq_length))
        self.token_control = torch.nn.Conv1d(seq_length, reduced_tokens, kernel_size=1)
        
        reduced_dims = int(math.sqrt(hidden_dim))
        self.channel_control = MLP(hidden_dim, [reduced_dims], activation_layer=nn.GELU, inplace=None, dropout=False)

        linear_dims = reduced_dims * reduced_tokens

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()

        if representation_size is None:
            heads_layers["head"] = nn.Linear(linear_dims, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(linear_dims, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        for i, _ in enumerate(self.patching_filters):
            if isinstance(self.patching_filters[i], nn.Conv2d):
                # Init the patchify stem
                fan_in = self.patching_filters[i].in_channels * self.patching_filters[i].kernel_size[0] * self.patching_filters[i].kernel_size[1]
                nn.init.trunc_normal_(self.patching_filters[i].weight, std=math.sqrt(1 / fan_in))
                if self.patching_filters[i].bias is not None:
                    nn.init.zeros_(self.patching_filters[i].bias)
            elif self.patching_filters[i].conv_last is not None and isinstance(self.patching_filters[i].conv_last, nn.Conv2d):
                # Init the last 1x1 conv of the conv stem
                nn.init.normal_(
                    self.patching_filters[i].conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.patching_filters[i].conv_last.out_channels)
                )
                if self.patching_filters[i].conv_last.bias is not None:
                    nn.init.zeros_(self.patching_filters[i].conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape

        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        
        multiscale_patched_input = [patching(x) for patching in self.patching_filters]
        multiscale_patched_input = [x.reshape(n, self.hidden_dim, seq_length) for (x, seq_length) in zip(multiscale_patched_input, self.sequence_lengths)]

        multiscale_patched_input = [x.permute(0,2,1) for x in multiscale_patched_input]
        
        x = torch.cat(multiscale_patched_input, dim=1) # concatenate along the sequence dimension
        
        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        x = self.encoder(x)

        x = self.token_control(x)

        x = self.channel_control(x)

        x = x.view(n, -1)

        x = self.heads(x)

        return x