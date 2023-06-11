# Base Vision Transformer Architecture Credit to PyTorch
# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.ops import MLP, Conv2dNormActivation

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU
        
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

class SpectralBlock(nn.Module):
    def __init__(
        self,
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

        self.weight_c = nn.Parameter(torch.empty(hidden_dim, hidden_dim, 2).normal_(std=0.02))  # from BERT
    
        #self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)

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

            x = torch.matmul(x, torch.view_as_complex(self.weight_c))

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
        for i, layer in enumerate(layer_config):
            if layer == 1:
                layers[f"spct_layer_{i}"] = SpectralBlock(
                    sequence_lengths,
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    norm_layer,
                )
            elif layer == 0:
                layers[f"atn_layer_{i}"] = AttentionBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
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
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
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
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.patching_filters = [nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=p, stride=p
            ).to(device) for p in patch_sizes]

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
        #p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        #n_h = h // p
        #n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        #x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        #x = x.reshape(n, self.hidden_dim, n_h * n_w)
        
        multiscale_patched_input = [patching(x) for patching in self.patching_filters]
        multiscale_patched_input = [x.reshape(n, self.hidden_dim, seq_length) for (x, seq_length) in zip(multiscale_patched_input, self.sequence_lengths)]

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        #x = x.permute(0, 2, 1)

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