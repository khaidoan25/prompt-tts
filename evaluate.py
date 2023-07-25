import argparse
import json
from tqdm import tqdm
from einops import rearrange

import torch
import torch.nn.functional as F

from diffusers import DDPMScheduler

from tts.models import (
    TTSSingleSpeaker, TTSMultiSpeaker,
    _samplewise_merge_tensors, list_to_tensor
)
from tts.dataloader.dataloader import create_dataloader
from tts.pipeline import MyPipeline


def eval_single(args):
    dataloader = create_dataloader(
        data_file=args.data_file,
        batch_size=1,
        max_seq_length=args.max_seq_length,
        shuffle=True,
        data_type="single_speaker"
    )
    
    config = json.load(open(args.config_file, "r"))
    model = TTSSingleSpeaker(config)
    
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    model.to("cuda")
    
    text_encoder = model.text_encoder
    unet = model.unet
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon"
    )
    
    decode_pipeline = MyPipeline(unet=unet, scheduler=noise_scheduler)
    
    total_loss = 0
    for batch in tqdm(dataloader):
        codes = batch["code"]
        text_seq_ids = batch["cmu_sequence_id"]
        attention_mask = batch["attention_mask"]
        
        text_emb = text_encoder(text_seq_ids.to("cuda"), attention_mask.to("cuda"))
        predicted_codes = decode_pipeline(text_emb, num_inference_steps=args.decode_step)
        
        loss = F.mse_loss(predicted_codes.cpu(), codes.cpu(), reduction="mean")
        
        total_loss += loss
        
    print("Avg loss: ", total_loss / len(dataloader))
    
    
def eval_multi(args):
    dataloader = create_dataloader(
        data_file=args.data_file,
        batch_size=1,
        max_seq_length=args.max_seq_length,
        shuffle=True,
        data_type="multi_speaker"
    )
    
    config = json.load(open(args.config_file, "r"))
    model = TTSMultiSpeaker(config)
    
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    model.to("cuda")
    
    text_encoder = model.text_encoder
    spk_encoder = model.spk_encoder
    unet = model.unet
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon"
    )
    
    decode_pipeline = MyPipeline(
        unet=unet,
        scheduler=noise_scheduler,
        in_channels=config["in_channels"],
        sample_size=config["sample_size"],
    )
    
    total_loss = 0
    for batch in tqdm(dataloader):
        codes = batch["code"]
        text_seq_ids = batch["cmu_sequence"] # List of Tensors
        spk_codes = batch["sample"] # List of Tensors
        
        # text_emb = text_encoder(text_seq_ids.to("cuda"))
        # spk_emb = spk_encoder(spk_codes.to("cuda"))
        # spk_emb = torch.permute(spk_emb, (0, 2, 1))
        # text_emb = torch.concat((text_emb, spk_emb), dim=1)
        
        text_seq_ids = [item.to("cuda") for item in text_seq_ids]
        text_emb_list = text_encoder(text_seq_ids)
        
        for i in range(len(spk_codes)):
            spk_codes[i] = rearrange(spk_codes[i], "l n -> n l").to("cuda")
        spk_emb_list = spk_encoder(spk_codes)
        
        input_list = _samplewise_merge_tensors(text_emb_list, spk_emb_list)
        
        x, m = list_to_tensor(input_list)
        
        predicted_codes = decode_pipeline(
            x.to("cuda"),
            attention_mask=m.to("cuda"),
            num_inference_steps=args.decode_step
        )
        
        loss = F.mse_loss(predicted_codes.cpu(), codes.cpu(), reduction="mean")
        
        total_loss += loss
        
    print("Avg loss: ", total_loss / len(dataloader))


def main(args):
    if args.data_type == "single_speaker":
        eval_single(args)
    elif args.data_type == "multi_speaker":
        eval_multi(args)
    else:
        assert f"{args.data_type} not implemented!!!"


def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate TTS model.")
    parser.add_argument('--data_file', type=str, default=None,
                        help="Path to the evaluation data file.", required=True)
    parser.add_argument('--config_file', type=str,
                        help="Path to config file.", required=True)
    parser.add_argument('--decode_step', type=int, default=100,
                        help="Diffusion decoding step.")
    parser.add_argument('--ckpt_path', type=str,
                        help="Directory to save checkpoints.", required=True)
    parser.add_argument('--max_seq_length', type=int, default=550,
                        help="Maximum length of cmu sequence.")
    parser.add_argument('--data_type', type=str, required=True)


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)