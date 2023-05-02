from argparse import ArgumentParser
from encodec import EncodecModel
import numpy as np
import torch
import soundfile as sf


model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)


def decode(encoded_frames: torch.Tensor):
    if len(encoded_frames.shape) != 3:
        raise BaseException("The encoded_frames must have the shape of [B, N_q, T]")
    
    return model.decode([(encoded_frames, None)])


def main(args):
    codec_matrix = np.load(args.npy_path)
    encoded_frames = torch.tensor(codec_matrix)
    
    if len(codec_matrix.shape) != 3:
        encoded_frames = encoded_frames.unsqueeze(0)
    
    with torch.no_grad():
        wav_dec = decode(encoded_frames)
    
    sf.write(
        args.npy_path.replace(".npy", ".wav"),
        wav_dec[0][0].numpy(),
        model.sample_rate
    )


def parse_args():
    parser = ArgumentParser(description="Test converting codec codes back to waveform.")
    
    parser.add_argument("--npy_path", required=True, help="Path to codec codes matrix.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    main(args)