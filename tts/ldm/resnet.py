from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.models.resnet import ResnetBlock2D, Downsample1D
from diffusers.models.embeddings import Timesteps, TimestepEmbedding



class Upsample1D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Parameters:
            channels: channels in the inputs and outputs.
            use_conv: a bool determining if a convolution is applied.
            use_conv_transpose:
            out_channels:
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x, output_size=None):
        assert x.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(x)

        if output_size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            x = F.interpolate(x, size=output_size, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x





class ResnetBlock1D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
            "ada_group" for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        skip_time_act=False,
        time_embedding_norm="default",  # default, scale_shift, ada_group
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        conv_shortcut_bias: bool = True,
        conv_1d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        # if self.time_embedding_norm == "ada_group":
        #     self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        # else:
        if self.time_embedding_norm == "default":
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group":
                self.time_emb_proj = None
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        # if self.time_embedding_norm == "ada_group":
        #     self.norm2 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        # else:
        if self.time_embedding_norm == "default":
            self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_1d_out_channels = conv_1d_out_channels or out_channels
        self.conv2 = torch.nn.Conv1d(out_channels, conv_1d_out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()

        self.upsample = self.downsample = None
        if self.up:
            # if kernel == "fir":
            #     fir_kernel = (1, 3, 3, 1)
            #     self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            # elif kernel == "sde_vp":
            #     self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            # else:
            self.upsample = Upsample1D(in_channels, use_conv=False)
        elif self.down:
            # if kernel == "fir":
            #     fir_kernel = (1, 3, 3, 1)
            #     self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            # elif kernel == "sde_vp":
            #     self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            # else:
            self.downsample = Downsample1D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            
            
    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        # if self.time_embedding_norm == "ada_group":
        #     hidden_states = self.norm1(hidden_states, temb)
        # else:
        if self.time_embedding_norm == "default":
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        # if self.time_embedding_norm == "ada_group":
        #     hidden_states = self.norm2(hidden_states, temb)
        # else:
        if self.time_embedding_norm == "default":
            hidden_states = self.norm2(hidden_states)

        # if temb is not None and self.time_embedding_norm == "scale_shift":
        #     scale, shift = torch.chunk(temb, 2, dim=1)
        #     hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


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
    
    # image
    # output_channel = block_out_channels[0]
    # input_channel = output_channel
    # model = ResnetBlock2D(
    #     in_channels=input_channel,
    #     out_channels=output_channel,
    #     temb_channels=block_out_channels[0] * 4,
    #     eps=1e-5,
    #     groups=norm_num_groups,
    #     dropout=0.0,
    #     time_embedding_norm="default",
    #     non_linearity=act_fn,
    #     output_scale_factor=1.0,
    #     pre_norm=True
    # )
    
    # sample = torch.rand(2, 4, 64, 64)
    # conv_in_kernel = 3
    # conv_in_padding = (conv_in_kernel - 1) // 2
    # conv_in = nn.Conv2d(
    #     in_channels, block_out_channels[0],
    #     kernel_size=conv_in_kernel, padding=conv_in_padding
    # )
    
    # sample = conv_in(sample)
    
    # sample.size()
    
    # output = model(sample, emb)
    
    # codec
    output_channel = block_out_channels[0]
    input_channel = output_channel
    model = ResnetBlock1D(
        in_channels=input_channel,
        out_channels=output_channel,
        temb_channels=block_out_channels[0] * 4,
        eps=1e-5,
        groups=norm_num_groups,
        dropout=0.0,
        time_embedding_norm="default",
        non_linearity=act_fn,
        output_scale_factor=1.0,
        pre_norm=True
    )
    
    sample = torch.rand(2, 8, 1500)
    conv_in_kernel = 3
    conv_in_padding = (conv_in_kernel -1 ) // 2
    conv_in = nn.Conv1d(
        in_channels, block_out_channels[0],
        kernel_size=conv_in_kernel, padding=conv_in_padding
    )
    
    sample = conv_in(sample)
    
    print(sample.size())
    
    # print(output.size())
    
