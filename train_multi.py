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

from tts.model.models import TTSMultiSpeaker
from tts.dataloader.dataloader import create_dataloader

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def test_main(args):
    config = json.load(open(args.config_file, "r"))
    
    model = TTSMultiSpeaker(config)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs]
    )
    
    # data
    dataloader = create_dataloader(
        args.data_files,
        args.batch_size,
        args.max_seq_length,
        data_type="multi_speaker",
        shuffle=True
    )
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / config["gradient_accumulation_steps"])
    max_train_steps = config["num_train_epochs"] * num_update_steps_per_epoch
    
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8
    )
        
    lr_scheduler = get_scheduler(
        config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"] * config["gradient_accumulation_steps"],
        num_training_steps=max_train_steps * config["gradient_accumulation_steps"],
    )
    
    dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        dataloader, model, optimizer, lr_scheduler
    )
    
    for epoch in range(config["num_train_epochs"]):
        epoch += args.prev_epoch
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for batch in pbar:
            with accelerator.accumulate(model):
                print("a")
        # End epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("This is main_process")
            print(epoch)
            print((epoch + 1) % config["save_per_epochs"])
            if (epoch + 1) % config["save_per_epochs"] == 0:
                print("Save model epoch")

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
    
    model = TTSMultiSpeaker(config)
        
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
        args.data_files,
        args.batch_size,
        args.max_seq_length,
        data_type="multi_speaker",
        use_tar=args.use_tar,
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
    
    if args.model_ckpt is not None and args.optim_ckpt is not None:
        # model.load_state_dict(
        #     torch.load(args.model_ckpt, map_location=accelerator.device)
        # )
        # optimizer.load_state_dict(
        #     torch.load(args.optim_ckpt, map_location=accelerator.device)
        # )
        accelerator.load_state(args.ckpt_dir)
    
    # training loop
    global_step = 0
    for epoch in range(config["num_train_epochs"]):
        epoch += args.prev_epoch
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        model.train()
        train_loss = 0.0
        
        for batch in pbar:
            with accelerator.accumulate(model):
                codes = batch["code"]
                text_seq_ids = batch["cmu_sequence"]
                spk_code = batch["sample"]
                code_length = batch["code_length"]
                
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
                
                output_unet, output_dp = model(
                    noisy_codes,
                    timesteps,
                    text_seq_ids,
                    spk_code
                ).sample
                
                unet_loss = F.mse_loss(output_unet.float(), noise.float(), reduction="mean")
                dp_loss = F.mse_loss(output_dp.float(), code_length, reduction="mean")
                
                loss = sum([unet_loss, dp_loss])
                
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
        
        # End epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrap_model = accelerator.unwrap_model(model)
            accelerator.save_state(output_dir=args.ckpt_dir)
            if (epoch + 1) % config["save_per_epochs"] == 0:
                accelerator.save(unwrap_model.state_dict(), args.ckpt_dir + f"ckpt_{epoch+1}.pt")
                accelerator.save(optimizer.state_dict(), args.ckpt_dir + f"optim_{epoch+1}.pt")
                # accelerator.save_state(output_dir=args.ckpt_dir)
            
    writer.flush()
    writer.close()
    
    
def parse_args():

    parser = argparse.ArgumentParser(description="Train TTS models."
                                     "The data is stored in WebDataset format.")
    parser.add_argument('--data_files', type=str, default=None,
                        help="Path to the training data files. Separate by ','.\
                            data_file1,data_file2",
                        required=True)
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
    parser.add_argument('--model_ckpt', type=str, default=None,
                        help="Path to the previous model checkpoint.")
    parser.add_argument('--optim_ckpt', type=str, default=None,
                        help="Path to the previous optimizer checkpoint.")
    parser.add_argument('--prev_epoch', type=int, default=0,
                        help="Num trained epoch.")
    parser.add_argument('--use_tar', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    