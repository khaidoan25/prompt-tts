import torch
from torch import nn
import numpy as np
from typing import Union, Dict, Any, Optional

from ldm.unet_1d_condition import Unet1DConditionModel

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