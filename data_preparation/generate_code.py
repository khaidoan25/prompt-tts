import argparse
import tarfile
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
from tempfile import TemporaryDirectory
import numpy as np
from os.path import abspath
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available else "cpu"
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)


def create_batch(members, tf, batch_size):
    index = 1
    batch = [[], []]
    for member in members:
        if ".wav" not in member.name:
            continue
        batch[1].append(member.name)
        wav, sr = torchaudio.load(tf.extractfile(member))
        if wav.shape[0] == 2:
            wav = wav[:1]
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        # Pad the wavfile to 20s duration
        wav = torch.cat([wav, torch.zeros((1, model.sample_rate*20-wav.shape[1]))], dim=-1)
        wav = wav.unsqueeze(0)
        batch[0].append(wav)
        if index % batch_size == 0:
            yield batch
            batch = [[], []]
        index += 1
    if len(batch) != 0:
        yield batch


def generate(batch):
    wav = torch.cat(batch).to(device)
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).cpu().numpy()
    
    return codes


def main(input_file, batch_size):
    output_file = input_file.replace(".tar", "_processed.tar")
    tf = tarfile.open(input_file, "r")
    members = tf.getmembers()
    print("Reading tarfile ...")
    
    output_tf = tarfile.open(output_file, "w")
    for batch in tqdm(create_batch(members, tf, batch_size)):
        codes = generate(batch[0])
        
        with TemporaryDirectory() as dirname:
            for i, code in enumerate(codes):
                np_file = batch[1][i].replace(".wav", ".npy").split('/')[-1]
                np.save(abspath(f"{dirname}/{np_file}"), code)
                output_tf.add(abspath(f"{dirname}/{np_file}"), arcname=np_file)
                
    # Move all text file to output tarfile
    with TemporaryDirectory() as dirname:
        for member in members:
            if ".txt" in member.name:
                tf.extract(member, dirname)
                suffix = member.name.split('/')[-1]
                output_tf.add(abspath(f"{dirname}/{member.name}"), arcname=suffix)
    
    tf.close()
    output_tf.close()

def parse_args():

    parser = argparse.ArgumentParser(description="Generate codec codes of waveform."
                                     "The data is stored in WebDataset format.")
    parser.add_argument('--input_file', type=str, default=None,
                        help="Path to the input file", required=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size of Encodec encode")


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args.input_file, args.batch_size)