import torch
import numpy as np
from einops import rearrange

from tts.utils import list_to_tensor, _samplewise_merge_tensors

from diffusers import DiffusionPipeline


class MyPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()

        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        text_seq_ids,
        spk_code,
        channels=8,
        noisy_codes=None,
        batch_size: int = 1,
        num_inference_steps: int = 50):
        
        # Predict duration
        duration = self.model.predict_duration(text_seq_ids)
        duration = int(np.ceil(duration.squeeze().cpu().numpy()))
        
        # Sample gaussian noise to begin loop
        if noisy_codes is None:
            noisy_codes = torch.randn(
                (
                    batch_size,
                    1,
                    duration,
                    channels,
                )
            )

        noisy_codes = noisy_codes.to("cuda")

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Prepare encoder hidden states
        text_seq_ids = [item.to("cuda") for item in text_seq_ids]
        text_emb_list = self.model.text_encoder(text_seq_ids) # list of [l, d]
        
        for i in range(len(spk_code)):
            spk_code[i] = rearrange(spk_code[i].squeeze(), "l n -> n l")
        spk_code = [item.to("cuda") for item in spk_code]
        spk_emb_list = self.model.spk_encoder(spk_code)

        input_list = _samplewise_merge_tensors(text_emb_list, spk_emb_list)
        
        encoder_hidden_states, _ = list_to_tensor(input_list)

        for t in self.progress_bar(self.scheduler.timesteps):
            
            # 1. predict noise model_output
            noise_pred = self.model.predict_unet(
                sample=noisy_codes,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            noisy_codes = self.scheduler.step(noise_pred, t, noisy_codes).prev_sample

        return noisy_codes