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
from io import BytesIO
import os


device = "cuda" if torch.cuda.is_available else "cpu"
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)


def create_batch(members, tf, batch_size, max_duration, ignore_list):            
    batch = [[], [], []]    # wav, name, codec_length
    for member in members:
        if ".wav" not in member.name:
            continue
        if member.name.split('/')[-1] in ignore_list:
            continue
        wav, sr = torchaudio.load(tf.extractfile(member))
        if wav.shape[0] == 2:
            wav = wav[:1]
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        if wav.shape[-1] > model.sample_rate * max_duration:
            print("Exceed max duration")
            print(member.name)
            continue
        # Pad the wavfile to 20s duration
        batch[2].append(np.ceil(wav.shape[1] / 320))
        wav = torch.cat(
            [wav, torch.zeros((1, model.sample_rate*max_duration-wav.shape[1]))],
            dim=-1
        )
        wav = wav.unsqueeze(0)
        batch[0].append(wav)
        batch[1].append(member.name)
        if len(batch[0]) % batch_size == 0 and len(batch[0]) != 0:
            yield batch
            batch = [[], [], []]
    if len(batch[0]) != 0:
        yield batch


def generate(batch):
    wav = torch.cat(batch).to(device)
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).cpu().numpy()
    
    return codes


class LibriTTSProcessor():
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        
        
    def standardize(
        self,
        ignore_file,
        batch_size=8,
        max_duration=12
        ):
        
        if os.path.exists(ignore_file):
            with open(ignore_file, "r") as f:
                ignore_list = set(f.read().split('\n')[:-1])
        else:
            with open(ignore_file, "w") as f:
                ignore_list = set([])
        
        print("Reading tarfile ...")
        orig_tf = tarfile.open(self.data_path, "r")
        members = orig_tf.getmembers()
        
        batch_generator = create_batch(
            members, 
            orig_tf, 
            batch_size, 
            max_duration,
            ignore_list
        )
        
        for batch in tqdm(batch_generator):
            if os.path.exists(self.data_path.replace(".tar", "_processed.tar")):
                open_type = "a"
            else:
                open_type = "w"
            
            codes = generate(batch[0])
            for i, code in enumerate(codes):
                try:
                    text_filename = batch[1][i].replace(".wav", ".original.txt")
                    text_file = orig_tf.getmember(text_filename)
                    text_norm_filename = batch[1][i].replace(".wav", ".normalized.txt")
                    text_norm_file = orig_tf.getmember(text_norm_filename)
                    
                    with TemporaryDirectory() as dirname:
                        np_file = batch[1][i].replace(".wav", ".npy").split('/')[-1]
                        np.save(abspath(f"{dirname}/{np_file}"), code)
                            
                        len_file = np_file.replace(".npy", ".len.txt")
                        with open(abspath(f"{dirname}/{len_file}"), "w") as f:
                            f.write(str(batch[2][i]))
                            
                        with tarfile.open(self.data_path.replace(".tar", "_processed.tar"), open_type) as new_tf:
                            new_tf.add(
                                abspath(f"{dirname}/{np_file}"),
                                arcname=batch[1][i].replace(".wav", ".npy")
                            )
                            new_tf.add(
                                abspath(f"{dirname}/{len_file}"),
                                arcname=batch[1][i].replace(".wav", ".len.txt")
                            )
                            new_tf.addfile(text_file)
                            new_tf.addfile(text_norm_file)
                        with open(ignore_file, "a") as f:
                            f.write(batch[1][i].split('/')[-1] + '\n')
                except:
                    print("Error occurs")
                    print(batch[1][i])
                    with open(ignore_file, "a") as f:
                        f.write(batch[1][i].split('/')[-1] + '\n')
                
        orig_tf.close()
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--ignore_file", required=True)
    parser.add_argument("--max_duration", default=20)
    parser.add_argument("--batch_size", default=16, type=int)
    args = parser.parse_args()
    
    data_processor = LibriTTSProcessor(
        args.input_file,
    )
    data_processor.standardize(
        args.ignore_file,
        max_duration=args.max_duration,
        batch_size=args.batch_size,
    )