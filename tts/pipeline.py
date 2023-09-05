import torch
import numpy as np
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
        attention_mask=None,
        channels=8,
        noisy_codes=None,
        batch_size: int = 1,
        num_inference_steps: int = 50):
        
        # Predict duration
        duration = self.model.predict_duration(text_seq_ids)
        duration = np.ceil(duration[0].cpu())
        
        # Sample gaussian noise to begin loop
        if noisy_codes is None:
            noisy_codes = torch.randn(
                (
                    batch_size,
                    channels,
                    duration
                )
            )

        noisy_codes = noisy_codes.to(self.device)
        # codes = codes * self.scheduler.init_noise_sigma

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            
            # 1. predict noise model_output
            noise_pred = self.model.predict_unet(
                sample=noisy_codes,
                timestep=t,
                text_seq_ids=text_seq_ids,
                spk_code=spk_code,
                attention_mask=attention_mask
            ).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            noisy_codes = self.scheduler.step(noise_pred, t, noisy_codes).prev_sample

        return noisy_codes