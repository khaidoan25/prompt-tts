import argparse
import pathlib
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


def create_batch(members, ignore_list, tf=None):
    if tf:
        for member in members:
            if ".wav" not in member.name:
                continue
            if member.name.split('/')[-1] in ignore_list:
                continue
            wav, sr = torchaudio.load(tf.extractfile(member))
            if wav.shape[0] == 2:
                wav = wav[:1]
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
            wav = wav.unsqueeze(0)
            
            yield wav, member.name, np.ceil(wav.shape[-1] / 320)
    else:
        for member in members:
            if member.split('/')[-1] in ignore_list:
                continue
            wav, sr = torchaudio.load(member)
            if wav.shape[0] == 2:
                wav = wav[:1]
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
            wav = wav.unsqueeze(0)
            
            yield wav, member, np.ceil(wav.shape[-1] / 320)


def generate(batch):
    wav = torch.tensor(batch).to(device)
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).cpu().numpy()
    
    return codes


def standardize_dir(args):
    if os.path.exists(args.ignore_file):
        with open(args.ignore_file, "r") as f:
            ignore_list = set(f.read().split('\n')[:-1])
    else:
        with open(args.ignore_file, "w") as f:
            ignore_list = set([])
            
    if not os.path.exists(args.error_file):
        with open(args.error_file, "w") as f:
            pass
        
    print("Reading directory ...")
    directory = pathlib.Path(args.input_file)
    
    wav_file = []
    for path in directory.rglob("*.wav"):
        wav_file.append(str(path))
        
    batch_generator = create_batch(
        wav_file,
        ignore_list,
        None,
    )
    
    for batch in tqdm(batch_generator):
        try:
            code = generate(batch[0])
            np.save(batch[1].replace(".wav", ".npy"), code)
            with open(batch[1].replace(".wav", ".len.txt"), "w") as f:
                f.write(str(batch[2]))
                
            with open(args.ignore_file, "a") as f:
                f.write(batch[1].split('/')[-1] + '\n')
        except:
            print("Error occurs")
            print(batch[1])
            with open(args.error_file, "a") as f:
                f.write(batch[1].split('/')[-1] + '\n')
    

def standardize_tar(args):
    if os.path.exists(args.ignore_file):
        with open(args.ignore_file, "r") as f:
            ignore_list = set(f.read().split('\n')[:-1])
    else:
        with open(args.ignore_file, "w") as f:
            ignore_list = set([])
            
    if not os.path.exists(args.error_file):
        with open(args.error_file, "w") as f:
            pass
    
    print("Reading tarfile ...")
    orig_tf = tarfile.open(args.data_path, "r:gz")
    members = orig_tf.getmembers()
    
    batch_generator = create_batch(
        members, 
        ignore_list,
        orig_tf,
    )
    
    for batch in tqdm(batch_generator):
        if os.path.exists(args.data_path.replace(".tar.gz", "_processed.tar")):
            open_type = "a"
        else:
            open_type = "w"
        
        code = generate(batch[0])
        try:
            text_filename = batch[1].replace(".wav", ".original.txt")
            text_file = orig_tf.getmember(text_filename)
            text_norm_filename = batch[1].replace(".wav", ".normalized.txt")
            text_norm_file = orig_tf.getmember(text_norm_filename)
            
            with TemporaryDirectory() as dirname:
                np_file = batch[1].replace(".wav", ".npy").split('/')[-1]
                np.save(abspath(f"{dirname}/{np_file}"), code)
                    
                len_file = np_file.replace(".npy", ".len.txt")
                with open(abspath(f"{dirname}/{len_file}"), "w") as f:
                    f.write(str(batch[2]))
                    
                with tarfile.open(args.data_path.replace(".tar.gz", "_processed.tar"), open_type) as new_tf:
                    new_tf.add(
                        abspath(f"{dirname}/{np_file}"),
                        arcname=batch[1].replace(".wav", ".npy")
                    )
                    new_tf.add(
                        abspath(f"{dirname}/{len_file}"),
                        arcname=batch[1].replace(".wav", ".len.txt")
                    )
                    new_tf.addfile(text_file)
                    new_tf.addfile(text_norm_file)
            with open(args.ignore_file, "a") as f:
                f.write(batch[1].split('/')[-1] + '\n')
        except:
            print("Error occurs")
            print(batch[1])
            with open(args.error_file, "a") as f:
                f.write(batch[1].split('/')[-1] + '\n')
            
    orig_tf.close()
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--ignore_file", required=True)
    parser.add_argument("--error_file", required=True)
    parser.add_argument(
        "--input_type",
        required=True,
        help="Type of the input file. 'dir' or 'tar'."    
    )
    args = parser.parse_args()
    
    if args.input_type == "tar":
        standardize_tar(args)
    elif args.input_type == "dir":
        standardize_dir(args)
    else:
        assert "Not implemented !!!"