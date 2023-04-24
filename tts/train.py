import logging
import json
import argparse
from tqdm import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import DDPMScheduler
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models import TTSSingleSpeaker
from dataloader import create_dataloader

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def main(args):
    writer = SummaryWriter(log_dir=args.log_dir)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    config = json.load(open(args.config_file, "r"))
    
    # data
    dataloader = create_dataloader(
        args.data_file,
        args.batch_size,
        args.max_seq_length
    )
    
    # scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon"
    )
    
    # model
    # model = Unet1DConditionModel(
    #     in_channels=config["in_channels"],
    #     out_channels=config["out_channels"],
    #     layers_per_block=config["layers_per_block"],
    #     block_out_channels=config["block_out_channels"],
    #     down_block_types=config["down_block_types"],
    #     up_block_types=config["up_block_types"],
    #     cross_attention_dim=config["cross_attention_dim"]
    # )
    model = TTSSingleSpeaker(config)
    
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8
    )
    
    mse = nn.MSELoss()
    
    dataloader, model, optimizer = accelerator.prepare(
        dataloader, model, optimizer
    )
    
    
    # training loop
    global_step = 0
    for epoch in range(100):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            codes = batch["code"]
            text_seq_ids = batch["cmu_sequence_id"]
            attention_mask = batch["attention_mask"]
            with accelerator.accumulate(model):
                noise = torch.randn(codes.shape).to(accelerator.device)
                bsz = codes.shape[0]
                
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=accelerator.device
                ).long()
                
                noisy_codes = noise_scheduler.add_noise(
                    codes, noise, timesteps
                )
                
                # print("codes: ", noisy_codes.size())
                # print("timesteps: ", timesteps.size())
                # print("text_seq_ids: ", text_seq_ids.size())
                # print("attention_mask: ", attention_mask.size())
                model_output = model(
                    noisy_codes,
                    timesteps,
                    text_seq_ids,
                    attention_mask,
                ).sample
                loss = mse(noise, model_output)
                
                writer.add_scalar("Loss/train", loss, global_step)
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pbar.set_postfix(MSE=loss.item())
            global_step += 1
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and (epoch + 1) % config["save_per_epochs"] == 0:
            unwrap_model = accelerator.unwrap_model(model)
            accelerator.save(unwrap_model.state_dict(), config["ckpt_dir"] + f"/ckpt_{epoch+1}.pt")
            accelerator.save(optimizer.state_dict(), config["ckpt_dir"] + f"/optim_{epoch+1}.pt")
            accelerator.save_state(output_dir=config["ckpt_dir"])
    writer.flush()
    writer.clos()
    
    
def parse_args():

    parser = argparse.ArgumentParser(description="Train TTS models."
                                     "The data is stored in WebDataset format.")
    parser.add_argument('--data_file', type=str, default=None,
                        help="Path to the training data file.", required=True)
    parser.add_argument('--log_dir', type=str,
                        help="Directory to save logs.", required=True)
    parser.add_argument('--config_file', type=str,
                        help="Path to config file.", required=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size of Encodec encode.")
    parser.add_argument('--max_seq_length', type=int, default=550,
                        help="Maximum length of cmu sequence.")


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    