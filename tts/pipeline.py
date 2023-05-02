import torch
from diffusers import DiffusionPipeline


class MyPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        encoder_hidden_states,
        noise=None,
        codes=None,
        batch_size: int = 1,
        num_inference_steps: int = 50):
        
        # Sample gaussian noise to begin loop
        if codes is None:
            codes = torch.randn(
                (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size
                )
            )

        codes = codes.to(self.device)
        # codes = codes * self.scheduler.init_noise_sigma

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            model_input = codes
            # model_input = self.scheduler.scale_model_input(model_input, t)
            
            # 1. predict noise model_output
            # noise_pred = self.unet(
            #     sample=model_input,
            #     timestep=t,
            #     encoder_hidden_states=encoder_hidden_states
            # ).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            codes = self.scheduler.step(noise, t, codes).prev_sample

        # x_0 = codes
        # codes = (codes / 2 + 0.5).clamp(0, 1)
        # codes = codes.cpu().numpy()
        # codes = (codes * 1023).round().astype('int')

        return codes