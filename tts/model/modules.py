import numpy as np
from typing import List

import torch
from torch import einsum, nn
import torch.nn.functional as F

from diffusers.models.attention import BasicTransformerBlock


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p_dropout):
        super().__init__()

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=kernel_size//2
        )
        self.norm_1 = LayerNorm(out_channels)
        self.conv_2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size, 
            padding=kernel_size//2
        )
        self.norm_2 = LayerNorm(out_channels)
        # self.conv_3 = nn.Conv1d(out_channels, 1, 1)
        self.proj = nn.Linear(out_channels, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)        # self.spk_encoder = nn

        x = self.conv_2(x * x_mask) # [b, out_channels, l]
        x = torch.sum(x, dim=-1) # [b, out_channels]
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x)
        return x
    
    
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
    
    
class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels
    
    
class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_len,
        seq_len,
        dim,
        attention_head_dim,
        dropout=0.0,
        num_layers=1
    ) -> None:
        super().__init__()
        
        self.word_embedding = nn.Embedding(vocab_len, dim)
        self.pos_embedding = PositionalEncodingPermute1D(seq_len)
        
        if dim % attention_head_dim == 0:
            num_attention_heads = dim // attention_head_dim
        else:
            raise "dim must be a multipliter of attention_head_dim"
            
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        
        
    def forward(self, input_ids, attention_mask):
        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(input_ids.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)        
        
        cmu_emb = self.word_embedding(input_ids)
        pos_emb = self.pos_embedding(cmu_emb)
        
        hidden_states = cmu_emb + pos_emb
        
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)
            
        return hidden_states
    
    
class TextEmbedding(nn.Embedding):
    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(x_list) == 0:
            return []
        return super().forward(torch.cat(x_list)).split([*map(len, x_list)])
    
    
class SpeakerEncoder(nn.Module):
    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        self.max_n_levels = max_n_levels
        self.n_tokens = n_tokens
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim)) # l k d
    
    def forward(
        self,
        x_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        w = self.weight

        padded_x_list = []

        for xi in x_list:
            xi = F.one_hot(xi, num_classes=self.n_tokens)  # n l k
            xi = xi.to(w)
            padded_x_list.append(xi)

        x = torch.cat(padded_x_list)  # n l k
        x = einsum("l k d, n l k -> n d", w, x)

        x_list = x.split([*map(len, x_list)])

        return x_list