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


device = "cuda" if torch.cuda.is_available else "cpu"
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)


def create_batch(members, tf, batch_size, max_duration):
    index = 1
    batch = [[], [], []]    # wav, name, codec_length
    for member in members:
        if ".wav" not in member.name:
            continue
        batch[1].append(member.name)
        wav, sr = torchaudio.load(tf.extractfile(member))
        if wav.shape[0] == 2:
            wav = wav[:1]
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        # Pad the wavfile to 20s duration
        batch[2].append(np.ceil(wav.shape[1] / 320))
        wav = torch.cat(
            [wav, torch.zeros((1, model.sample_rate*max_duration-wav.shape[1]))],
            dim=-1
        )
        wav = wav.unsqueeze(0)
        batch[0].append(wav)
        if index % batch_size == 0:
            yield batch
            batch = [[], [], []]
        index += 1
    if len(batch) != 0:
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
        batch_size=8,
        max_duration=12
        ):
        orig_tf = tarfile.open(self.data_path, "r:gz")
        members = orig_tf.getmembers()
        print("Reading tarfile ...")
        
        new_tf = tarfile.open(self.data_path.replace(".gz", ""), "w")
        
        for batch in tqdm(create_batch(members, orig_tf, batch_size, max_duration)):
            codes = generate(batch[0])

            for i, code in enumerate(codes):
                with TemporaryDirectory() as dirname:
                    # save code
                    np_file = batch[1][i].replace(".wav", ".npy").split('/')[-1]
                    np.save(abspath(f"{dirname}/{np_file}"), code)
                    new_tf.add(abspath(f"{dirname}/{np_file}"), arcname=np_file)
                    
                    # save length
                    len_file = np_file.replace(".npy", ".len.txt")
                    with open(abspath(f"{dirname}/{len_file}"), "w") as f:
                        f.write(str(batch[2][i]))
                    new_tf.add(abspath(f"{dirname}/{len_file}"), arcname=len_file)
                    
                # copy transcript
                orig_trans = batch[1][i].replace(".wav", ".original.txt")
                norm_trans = batch[1][i].replace(".wav", ".normalized.txt")
                
                text = orig_tf.extractfile(orig_trans).read()
                text_norm = orig_tf.extractfile(norm_trans).read()
                
                string = BytesIO(bytes(text, "utf-8"))
                tarinfo = tarfile.TarInfo(orig_trans.split('/')[-1])
                tarinfo.size = len(text)
                new_tf.addfile(tarinfo, string)
                
                string_norm = BytesIO(bytes(text_norm, "utf-8"))
                tarinfo_norm = tarfile.TarInfo(norm_trans.split('/')[-1])
                tarinfo_norm.size = len(text_norm)
                new_tf.addfile(tarinfo_norm, string_norm)
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    args = parser.parse_args()
    
    data_processor = LibriTTSProcessor(args.input_file)
    data_processor.standardize()