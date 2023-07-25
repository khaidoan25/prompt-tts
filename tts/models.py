from einops import rearrange
from functools import partial
import numpy as np
from typing import Union, Dict, Any, Optional, List, Tuple

import torch
from torch import einsum, nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tts.ldm.unet_1d_condition import Unet1DConditionModel

from diffusers.models.attention import BasicTransformerBlock


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
    
    
class TTSSingleSpeaker(nn.Module):
    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_len=config["cmu_vocab_len"],
            seq_len=config["cmu_seq_len"],
            dim=config["cross_attention_dim"],
            attention_head_dim=config["attention_head_dim"],
            dropout=config["text_encoder_dropout"],
            num_layers=config["text_encoder_layers"],
        )
        
        self.unet = Unet1DConditionModel(
            sample_size=config["sample_size"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            layers_per_block=config["layers_per_block"],
            block_out_channels=config["block_out_channels"],
            down_block_types=config["down_block_types"],
            mid_block_type=config["mid_block_type"],
            up_block_types=config["up_block_types"],
            cross_attention_dim=config["cross_attention_dim"],
        )
        
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        text_seq_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]]=None,
        return_dict: bool=True,
    ):
        text_emb = self.text_encoder(text_seq_ids, attention_mask)
        
        # print("text_emb: ", text_emb.size())
        
        unet_output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=text_emb,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict
        )
        
        return unet_output
    
    
class Embedding(nn.Embedding):
    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        print()
        if len(x_list) == 0:
            return []
        return super().forward(torch.cat(x_list)).split([*map(len, x_list)])
    
    
def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
    stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
    return (seq < stop).float()  # (b t)


def list_to_tensor(x_list: List[torch.Tensor], pattern="n b d -> b n d"):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    l = list(map(len, x_list))
    x = rearrange(pad_sequence(x_list), pattern)
    m = _create_mask(l, x_list[0].device)
    m = m.t().unsqueeze(-1)  # (t b 1)
    m = rearrange(m, pattern)
    m = m.to(x)
    return x, m


def _join(x: Tuple[torch.Tensor], sep: torch.Tensor):
    """
    Args:
        x: (k t d)
        sep: (d)
    """
    ret = x[0]
    for i in range(1, len(x)):
        ret = torch.cat((ret, sep[None], x[i]), dim=0)
    return ret


def _samplewise_merge_tensors(*l, sep: Optional[torch.Tensor]=None):
    if sep is None:
        cat = torch.cat
    else:
        cat = partial(_join, sep=sep)
    return [*map(cat, zip(*l))]
    

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
    
    
class TTSMultiSpeaker(nn.Module):
    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.text_encoder = Embedding(
            config["cmu_vocab_len"],
            config["cross_attention_dim"]
        )
        
        self.unet = Unet1DConditionModel(
            sample_size=config["sample_size"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            layers_per_block=config["layers_per_block"],
            block_out_channels=config["block_out_channels"],
            down_block_types=config["down_block_types"],
            mid_block_type=config["mid_block_type"],
            up_block_types=config["up_block_types"],
            cross_attention_dim=config["cross_attention_dim"],
        )
        
        # self.spk_encoder = nn.Conv1d(
        #     8,
        #     config["cross_attention_dim"],
        #     kernel_size=5,
        # )
        self.spk_encoder = SpeakerEncoder(
            8,
            1024,
            config["cross_attention_dim"]
        )
        
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        text_seq_ids: List[torch.Tensor],
        spk_code: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor]=None,
        cross_attention_kwargs: Optional[Dict[str, Any]]=None,
        return_dict: bool=True,
    ):
        text_emb_list = self.text_encoder(text_seq_ids)
        
        for i in range(len(spk_code)):
            spk_code[i] = rearrange(spk_code[i], "l n -> n l")
        spk_emb_list = self.spk_encoder(spk_code)

        input_list = _samplewise_merge_tensors(text_emb_list, spk_emb_list)
        
        x, m = list_to_tensor(input_list)
        
        unet_output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=x,
            attention_mask=m,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict
        )
        
        return unet_output
    
    
if __name__ == "__main__":
    encoder = TextEncoder(
        145,
        512,
        768,
        8,
        96
    )
    
    input_ids = torch.rand(2, 512).long()
    attention_mask = torch.rand(2, 512)
    
    hidden_states = encoder(input_ids, attention_mask)
    
    print(hidden_states.size())