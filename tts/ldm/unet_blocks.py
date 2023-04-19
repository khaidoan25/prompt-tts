import torch
from torch import nn

from resnet import ResnetBlock1D
from transformer_1d import Transformer1DModel

from diffusers.models.embeddings import Timesteps, TimestepEmbedding


class CrossAttnDownBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Transformer1DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
            
    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None
    ):
        output_states = ()
        
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            print(hidden_states.size())
            print(encoder_hidden_states.size())
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states,
                cross_attention_kwargs
            )[0]
            
            output_states += (hidden_states, )
            
        return hidden_states, output_states
            
            
if __name__ == "__main__":
    import json
    
    with open("/home/khaidoan/projects/def-mageed/khaidoan/speech/work_dir/prompt-tts/run_code/1d_config.json", "r") as f:
        config = json.load(f)
        
    in_channels = config["in_channels"]
    block_out_channels = config["block_out_channels"]
    attention_head_dim = config["attention_head_dim"]
    act_fn = config["act_fn"]
    norm_num_groups = config["norm_num_groups"]
    flip_sin_to_cos = config["flip_sin_to_cos"]
    freq_shift = config["freq_shift"]
    cross_attention_dim = config["cross_attention_dim"]
    
    # time
    time_embed_dim = block_out_channels[0] * 4
    timestep_input_dim = block_out_channels[0]
    time_proj = Timesteps(
        block_out_channels[0],
        flip_sin_to_cos,
        freq_shift
    )
    time_embedding = TimestepEmbedding(
        timestep_input_dim,
        time_embed_dim,
        act_fn=act_fn,
        post_act_fn=None,
        cond_proj_dim=None,
    )
    
    bsz = 2
    timesteps = torch.randint(0, 10, (bsz, ), device="cpu")
    timesteps = timesteps.long()
    
    timesteps = timesteps.expand(bsz)
    t_emb = time_proj(timesteps)
    emb = time_embedding(t_emb, None)
    
    output_channel = block_out_channels[0]
    input_channel = output_channel
    model = CrossAttnDownBlock1D(
        in_channels=input_channel,
        out_channels=output_channel,
        temb_channels=block_out_channels[0] * 4,
        cross_attention_dim=cross_attention_dim,
    )
    
    sample = torch.rand(2, 8, 1500)
    conv_in_kernel = 3
    conv_in_padding = (conv_in_kernel -1 ) // 2
    conv_in = nn.Conv1d(
        in_channels, block_out_channels[0],
        kernel_size=conv_in_kernel, padding=conv_in_padding
    )
    
    sample = conv_in(sample)
    condition = torch.rand(2, 50, cross_attention_dim)
    
    hidden_states, res_hidden_states = model(sample, emb, condition)
    
    print(hidden_states.size())