import sys
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
import numpy as np
import copy
from accelerate import DistributedDataParallelKwargs
import transformers
from multilingual_clip import pt_multilingual_clip
import argparse
import inspect
import math
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.deprecation_utils import deprecate
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel,UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version,BaseOutput
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from process_text import text_to_sequence, cmudict, sequence_to_text
from process_text.symbols import symbols
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print('not mps')
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


class MyPipe(DDPMPipeline):
    def __init__(self, unet, scheduler):
        super(MyPipe,self).__init__(unet, scheduler)
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        arp=None,
        attention_mask=None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        message = (
            "Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
            " DDPMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`."
        )
        predict_epsilon = deprecate("predict_epsilon", "0.13.0", message, take_from=kwargs)

        if predict_epsilon is not None:
            new_config = dict(self.scheduler.config)
            new_config["prediction_type"] = "epsilon" if predict_epsilon else "sample"
            self.scheduler._internal_dict = FrozenDict(new_config)

        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `torch.Generator(device="{self.device}")` instead.'
            )
            deprecate(
                "generator.device == 'cpu'",
                "0.13.0",
                message,
            )
            generator = None

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t,arp=arp,attention_mask=attention_mask).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 1023).round().astype('int')
        image = np.squeeze(image, axis=3)
        return image

class UNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor


class MyModel(UNet2DConditionModel):
    

    
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: str = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        total_vocab=150,
        total_length=256
    ):
        super(MyModel,self).__init__(sample_size,
                                    in_channels,
                                    out_channels,
                                    center_input_sample,
                                    flip_sin_to_cos,
                                    freq_shift,
                                    down_block_types,
                                    mid_block_type,
                                    up_block_types,
                                    only_cross_attention,
                                    block_out_channels,
                                    layers_per_block,
                                    downsample_padding,
                                    mid_block_scale_factor,
                                    act_fn,
                                    norm_num_groups,
                                    norm_eps,
                                    cross_attention_dim,
                                    attention_head_dim,
                                    dual_cross_attention,
                                    use_linear_projection,
                                    class_embed_type,
                                    num_class_embeds,
                                    upcast_attention,
                                    resnet_time_scale_shift)
        
        self.word_embedding=torch.nn.Embedding(total_vocab, cross_attention_dim)
        self.position_embeddings = torch.nn.Embedding(total_length, cross_attention_dim)
        self.register_buffer("position_ids", torch.arange(total_length).expand((1, -1)))
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        arp: torch.Tensor,
        attention_mask: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        #attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            #logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        
        arp_emb=self.word_embedding(arp)
        
        position_ids = self.position_ids[:, : 256 ]
        pos_emb = self.position_embeddings(position_ids)
        encoder_hidden_states=arp_emb+pos_emb
        
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
    
def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    print(accelerator.state.num_processes)
    
    model=MyModel(sample_size=(8,1500),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128,256, 512, 512),
            down_block_types=(

                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
         
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",


            ),
            cross_attention_dim=768,
    )

    train= pd.read_csv('lj.csv')
     



    noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )

    cmu_dict = cmudict.CMUDict("process_text/cmu_dictionary")
    def intersperse(lst, item):
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result



    class MyDataset():
        def __init__(self,df):
            self.raw = list(df.text)
            self.paths = list(df.path)
            self.normalization=torchvision.transforms.Normalize(0.5, 0.5)
        def __len__(self): return len(self.raw)

        def __getitem__(self,idx):
            codecs = np.load('LJSpeech/'+self.paths[idx])
            codecs=torch.from_numpy(codecs)
            codecs=codecs/1023
            codecs=torch.unsqueeze(codecs, dim=0)
            codecs=self.normalization(codecs)
            cmu_sequence = intersperse(
                text_to_sequence(self.raw[idx], ["english_cleaners"], cmu_dict),
                len(symbols)
            )
            return codecs,cmu_sequence

    def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
        result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
        mask_ = torch.full([len(examples), max_length], 0, dtype=torch.int64).tolist()
        for i, example in enumerate(examples):
            curr_len = min(len(example), max_length)
            result[i][:curr_len] = example[:curr_len]
            mask_[i][:curr_len] = [1] * curr_len
        if return_mask:
            return result, mask_
        return result

    def pad_TextSequence(batch):
        return torch.nn.utils.rnn.pad_sequence(batch,batch_first=True, padding_value=0)

    def collate_fn(batch):
        codes = []
        attn=[]

        for x,y in batch:
            codes += [x]
            attn += [y]
    
        attns,mask = _collate_batch_helper(attn,149,256,return_mask=True)
        attns=torch.IntTensor(attns)
        mask=torch.IntTensor(mask)
        codes=torch.cat(codes, dim=0)
        return  codes,attns,mask




    train_ds = MyDataset(train)
    tr_dataloader=torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True,collate_fn=collate_fn)



    def get_concat_h(im_list):
            dst = Image.new('RGB', (im_list[0].width*5, im_list[0].height*3))
            for i in range(5):
                dst.paste(im_list[i], (im_list[0].width*i, 0))
            for i in range(5,10):
                dst.paste(im_list[i], (im_list[0].width*(i-5), im_list[0].height))
            for i in range(10,15):
                dst.paste(im_list[i], (im_list[0].width*(i-10), im_list[0].height*2))
            return dst




        

    def get_generate_images(mod,encoder_hidden_states):
            pipeline = MyPipe(
                            unet=mod,
                            scheduler=noise_scheduler,
                    )
            generator = torch.Generator(device=pipeline.device).manual_seed(0)
            images = pipeline(
                    generator=generator,
                    batch_size=3,
                    encoder_hidden_states=encoder_hidden_states
                    )
            return images


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    mse = nn.MSELoss()
    tr_dataloader, model, optimizer = accelerator.prepare(
        tr_dataloader, model,optimizer
    )
    
    
    for epoch in range(0,3000):

        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(tr_dataloader, disable=not accelerator.is_local_main_process)
        for i, (images, arp,attention_mask) in enumerate(pbar):
            with accelerator.accumulate(model):

                images=torch.unsqueeze(images, dim=1)
                noise = torch.randn(images.shape).to(accelerator.device)
                bsz = images.shape[0]

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
                ).long()
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
               
                model_output = model(noisy_images, timesteps,arp=arp,attention_mask=attention_mask).sample
                loss = mse(noise, model_output)
            
            
            
            

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pbar.set_postfix(MSE=loss.item())
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrap_model = accelerator.unwrap_model(model)
            accelerator.save(unwrap_model.state_dict(), "/l/users/qisheng.liao/t2s/second/models/ckpt.pt")
            accelerator.save(optimizer.state_dict(), "/l/users/qisheng.liao/t2s/second/models/optim.pt")
            accelerator.save_state(output_dir="/l/users/qisheng.liao/t2s/second/my_checkpoint")


if __name__ == "__main__":
    main()
