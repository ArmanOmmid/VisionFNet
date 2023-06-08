
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.ops import MLP
        
class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        seq_length: int,
        config : object,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.config = config

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)

        # self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.L = seq_length if not self.config.class_token else seq_length - 1
        self.H = self.W = int(math.sqrt(self.L))
        self.F = int(self.W // 2) + 1
        self.mixer = nn.Parameter(torch.empty(self.H, self.F, hidden_dim, hidden_dim, 2, dtype=torch.float32).normal_(std=0.02))

        self.G = seq_length
        self.in_dims = (hidden_dim // self.num_heads)# * 2 NOT complex
        self.QK_d = self.in_dims
        self.V_d = self.in_dims

        # self.injecter = MLP(hidden_dim, int(math.sqrt(hidden_dim)), activation_layer=nn.GELU, inplace=None, dropout=dropout)

        # Q is on a vector
        self.Q_w = nn.Parameter(torch.empty(self.QK_d, self.num_heads, self.in_dims, dtype=torch.float32).normal_(std=0.02))
        self.Q_b = nn.Parameter(torch.empty(1, self.num_heads, 1, dtype=torch.float32).normal_(std=0.02))

        self.K_w = nn.Parameter(torch.empty(self.QK_d, self.num_heads, self.in_dims, dtype=torch.float32).normal_(std=0.02))
        self.K_b = nn.Parameter(torch.empty(self.G, self.num_heads, 1, dtype=torch.float32).normal_(std=0.02))

        self.V_w = nn.Parameter(torch.empty(self.V_d, self.num_heads, self.in_dims, dtype=torch.float32).normal_(std=0.02))
        self.V_b = nn.Parameter(torch.empty(self.G, self.num_heads, 1, dtype=torch.float32).normal_(std=0.02))

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(hidden_dim, [mlp_dim, hidden_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        x = self.ln_1(input)

        N, L, C = input.shape
        if self.config.class_token:
            class_token, x = input[:, 0].reshape(N, 1, C), input[:, 1:]
            L = L - 1

        H = W = int(math.sqrt(L))
        F = int(W // 2) + 1 # Fourier Width
        G = L + 1


        x = x.view(N, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        mixer = torch.view_as_complex(self.mixer)
        x = torch.einsum("nhfd,hfds->nhfd", x, mixer)

        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.reshape(N, L, C)

        if self.config.class_token:
            
            full = torch.cat((class_token, x), axis=1)

            Q = class_token.view(N, 1, self.num_heads, self.QK_d)
            K = full.view(N, G, self.num_heads, self.QK_d)
            V = full.view(N, G, self.num_heads, self.V_d)

            Q = torch.einsum("nqhd,xhd->nqhx", Q, self.Q_w) + self.Q_b
            K = torch.einsum("nkhd,xhd->nkhx", K, self.K_w) + self.K_b
            V = torch.einsum("nvhd,xhd->nvhx", V, self.V_w) + self.V_b

            A = torch.einsum("nqhd,nkhd->nhqk", Q, K) # q and k are the lengths which equal g. d represents the q and k dims
            A = torch.softmax(A / (self.QK_d ** 0.5), dim=3)

            new_class_token = torch.einsum("nhqk,nkhd->nqhd", A, V)
            new_class_token = new_class_token.reshape(N, 1, C)
            x = torch.cat((new_class_token, x), axis=1)

        # x, _ = self.self_attention(x, x, x, need_weights=False)
            
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)

        return x + y

class Encoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        config: object,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                seq_length,
                config,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        x = self.layers(self.dropout(input))
        x = self.ln(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        config: object,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.config = config

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        linear_size = hidden_dim
        if self.config.class_token:
            # Add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1
        else:
            linear_size = hidden_dim * seq_length

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            config,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(linear_size, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(linear_size, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        if self.config.class_token:
            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        if self.config.class_token:
            # Classifier "token" as used by standard language architectures
            x = x[:, 0]
        else:
            x = x.view(n, -1)

        x = self.heads(x)

        return x