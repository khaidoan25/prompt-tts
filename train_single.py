import logging
import json
import argparse
import math
from tqdm import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tts.models import TTSSingleSpeaker
from tts.dataloader.dataloader import create_dataloader

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def main(args):
    writer = SummaryWriter(log_dir=args.log_dir)
    config = json.load(open(args.config_file, "r"))
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs]
    )
    
    # scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon"
    )
    
    model = TTSSingleSpeaker(config)
        
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8
    )
    
    # data
    dataloader = create_dataloader(
        args.data_file,
        args.batch_size,
        args.max_seq_length,
        shuffle=True
    )
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / config["gradient_accumulation_steps"])
    max_train_steps = config["num_train_epochs"] * num_update_steps_per_epoch
        
    lr_scheduler = get_scheduler(
        config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"] * config["gradient_accumulation_steps"],
        num_training_steps=max_train_steps * config["gradient_accumulation_steps"],
    )
    
    dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        dataloader, model, optimizer, lr_scheduler
    )
    
    # training loop
    global_step = 0
    for epoch in range(config["num_train_epochs"]):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        model.train()
        train_loss = 0.0
        
        for batch in pbar:
            with accelerator.accumulate(model):
                codes = batch["code"]
                text_seq_ids = batch["cmu_sequence_id"]
                attention_mask = batch["attention_mask"]
                
                # Sample noise
                noise = torch.randn_like(codes).to(accelerator.device)
                
                bsz = codes.shape[0]
                
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=accelerator.device
                ).long()
                
                # Add noise
                noisy_codes = noise_scheduler.add_noise(
                    codes, noise, timesteps
                )
                
                model_pred = model(
                    noisy_codes,
                    timesteps,
                    text_seq_ids,
                    attention_mask,
                ).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / config["gradient_accumulation_steps"]
                
                writer.add_scalar("Loss/train", train_loss, global_step)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # End step
            pbar.set_postfix(MSE=train_loss)
            global_step += 1
            train_loss = 0.0
        
        # Save for debugging
        # torch.save(noise[:1], "noise.pt")
        # torch.save(codes[:1], "x_0.pt")
        # torch.save(text_seq_ids[:1], "text_seq_ids.pt")
        # torch.save(attention_mask[:1], "attn_mask.pt")
        # torch.save(timesteps[:1], "timesteps.pt")
        # torch.save(noisy_codes[:1], "x_t.pt")
        # with open("same_test.txt", "w") as f:
        #     f.write(batch["text"][0])
        
        
        # End epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and epoch % config["save_per_epochs"] == 0:
            unwrap_model = accelerator.unwrap_model(model)
            accelerator.save(unwrap_model.state_dict(), args.ckpt_dir + f"ckpt_{epoch+1}.pt")
            accelerator.save(optimizer.state_dict(), args.ckpt_dir + f"optim_{epoch+1}.pt")
            accelerator.save_state(output_dir=args.ckpt_dir)
    accelerator.end_training()
    writer.flush()
    writer.close()
    
    
def parse_args():

    parser = argparse.ArgumentParser(description="Train TTS models."
                                     "The data is stored in WebDataset format.")
    parser.add_argument('--data_file', type=str, default=None,
                        help="Path to the training data file.", required=True)
    parser.add_argument('--log_dir', type=str,
                        help="Directory to save logs.", required=True)
    parser.add_argument('--config_file', type=str,
                        help="Path to config file.", required=True)
    parser.add_argument('--ckpt_dir', type=str,
                        help="Directory to save checkpoints.", required=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size of Encodec encode.")
    parser.add_argument('--max_seq_length', type=int, default=550,
                        help="Maximum length of cmu sequence.")


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    