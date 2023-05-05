import argparse
import json
import numpy as np

import torch

from diffusers import DDPMScheduler

from tts.models import TTSSingleSpeaker
from tts.utils import get_cwd, transform_to_code
from tts.process_text import text_to_sequence, cmudict, sequence_to_text
from tts.process_text.symbols import symbols
from tts.dataloader import intersperse
from tts.pipeline import MyPipeline


def prepare_input(text, pad_token_id=0):
    cwd = get_cwd()
    cmu_dict = cmudict.CMUDict(cwd + "/tts/process_text/cmu_dictionary")
    
    cmu_sequence = intersperse(
        text_to_sequence(text, ["english_cleaners"], cmu_dict),
        len(symbols)
    )
    mask = torch.full([1, len(cmu_sequence)], 1, dtype=torch.int64)
    
    return torch.IntTensor(cmu_sequence).unsqueeze(0), mask


def main(args):
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
    
    cmu_seq_ids, attn_mask = prepare_input(args.text)
    
    text_emb = text_encoder(cmu_seq_ids.to("cuda"), attn_mask.to("cuda"))
    
    decode_pipeline = MyPipeline(unet=unet, scheduler=noise_scheduler)
    
    normalized_codes = decode_pipeline(text_emb, num_inference_steps=args.decode_step)
    
    codes = transform_to_code(normalized_codes)
    
    return codes


def parse_args():

    parser = argparse.ArgumentParser(description="Train TTS models."
                                     "The data is stored in WebDataset format.")
    parser.add_argument('--text', type=str,
                        help="Text sample to run TTS", required=True)
    parser.add_argument('--config_file', type=str,
                        help="Path to config file.", required=True)
    parser.add_argument('--decode_step', type=int, default=100,
                        help="Diffusion decoding step.")
    parser.add_argument('--ckpt_path', type=str,
                        help="Path to saved checkpoints.", required=True)
    parser.add_argument('--max_seq_length', type=int, default=550,
                        help="Maximum length of cmu sequence.")


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)