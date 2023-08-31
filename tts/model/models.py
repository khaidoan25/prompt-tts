from einops import rearrange
from functools import partial
import numpy as np
from typing import Union, Dict, Any, Optional, List

import torch
from torch import nn

from tts.ldm.unet_1d_condition import Unet1DConditionModel
from tts.model.modules import (
    TextEncoder, TextEmbedding,
    SpeakerEncoder, DurationPredictor
)
from tts.utils import list_to_tensor, _samplewise_merge_tensors
    
    
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
    
    
class TTSMultiSpeaker(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEmbedding(
            config["cmu_vocab_len"],
            config["cross_attention_dim"]
        )
        
        self.duration_predictor = DurationPredictor(
            in_channels=config["cross_attention_dim"],
            out_channels=config["out_channels_dp"],
            kernel_size=config["kernel_size"],
            p_dropout=config["p_dropout"]
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
        cross_attention_kwargs: Optional[Dict[str, Any]]=None,
        return_dict: bool=True,
    ):
        text_emb_list = self.text_encoder(text_seq_ids) # list of [l, d]
        
        x_dp, m_dp = list_to_tensor(text_emb_list) # [b, l, d], [b, l, 1]
        x_dp = torch.transpose(x_dp, 1, -1)
        x_dp = torch.detach(x_dp)
        m_dp = torch.transpose(m_dp, 1, -1)
        
        #TODO:
        # 1. Finish the flow with duration predictor (in review)
        # 2. Define a function to predict duration (in review)
        # 3. Define a cuntion to get encoder hidden states (in review)
        dp_output = self.duration_predictor(x_dp, m_dp)
        
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
        
        return unet_output, dp_output
    
    def predict_duration(
        self,
        text_seq_ids: List[torch.Tensor]
    ):
        text_emb_list = self.text_encoder(text_seq_ids) # list of [l, d]
        
        x_dp, m_dp = list_to_tensor(text_emb_list) # [b, l, d], [b, l, 1]
        x_dp = torch.transpose(x_dp, 1, -1)
        x_dp = torch.detach(x_dp)
        m_dp = torch.transpose(m_dp, 1, -1)
        
        dp_output = self.duration_predictor(x_dp, m_dp)
        return dp_output
    
    def predict_unet(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        text_seq_ids: List[torch.Tensor],
        spk_code: List[torch.Tensor],
        cross_attention_kwargs: Optional[Dict[str, Any]]=None,
        return_dict: bool=True
    ):
        text_emb_list = self.text_encoder(text_seq_ids) # list of [l, d]
        
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
    