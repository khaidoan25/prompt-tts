import pathlib
import tarfile
import random
import numpy as np
from io import BytesIO, TextIOWrapper

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

from tts.utils import get_cwd
from tts.process_text import text_to_sequence, cmudict, sequence_to_text
from tts.process_text.symbols import symbols


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_speaker_id(file_name):
    file_name = file_name.split('/')[-1]
    speaker_id = file_name.split('_')[0]
    
    return speaker_id
    
    
class MultiSpeakerDataset(Dataset):
    def __init__(self, data_paths, use_tar=True) -> None:
        super().__init__()
        self.normalization = torchvision.transforms.Normalize([0.5], [0.5])
        self.speaker_dict = {}
        self.item_list = []
        
        if use_tar:
            self.prepare_data_tarfile(data_paths)
        else:
            self.prepare_data_dir(data_paths)
        
    def prepare_data_dir(self, data_paths):
        cwd = get_cwd()    
        cmu_dict = cmudict.CMUDict(cwd + "/tts/process_text/cmu_dictionary")
        
        ignore_file = set() # Avoid identical data points
        data_paths = data_paths.split(',')
        for data_path in data_paths:
            print("Prepare dataset (directory):")
            print(data_path)
            directory = pathlib.Path(data_path)
            
            npy_file = []
            for path in directory.rglob("*.npy"):
                npy_file.append(str(path))
            txt_file = []
            for path in directory.rglob("*.txt"):
                txt_file.append(str(path))
                
            for codec_file in npy_file:
                if codec_file in ignore_file:
                    continue
                codec_code = np.load(codec_file)
                
                with open(codec_file.replace(".npy", ".original.txt"), "r") as f:
                    text = f.read()
                
                if codec_file.replace(".npy", ".normalized.txt") not in txt_file:
                    text_norm = text
                else:
                    with open(codec_file.replace(".npy", ".normalized.txt"), "r") as f:
                        text_norm = f.read()
                        
                if len(text_norm) == 0:
                    continue
                
                cmu_sequence = intersperse(
                    text_to_sequence(text_norm, ["english_cleaners"], cmu_dict),
                    len(symbols)
                )

                with open(codec_file.replace(".npy", ".len.txt"), "r") as f:
                    length = int(float(f.read()))
                    
                speaker_id = get_speaker_id(codec_file)
                if speaker_id not in self.speaker_dict.keys():
                    self.speaker_dict[speaker_id] = [codec_code]
                else:
                    self.speaker_dict[speaker_id].append(codec_code)
                    
                self.item_list.append(
                    {
                        "code": codec_code / 1023,
                        "ignore_code": codec_code,
                        "text": text,
                        "text_norm": text_norm,
                        "cmu_sequence": cmu_sequence,
                        "code_length": length,
                        "speaker_id": speaker_id
                    }
                )
            ignore_file.update(codec_file)
    
    def prepare_data_tarfile(self, data_paths):
        cwd = get_cwd()    
        cmu_dict = cmudict.CMUDict(cwd + "/tts/process_text/cmu_dictionary")
        
        ignore_file = set() # Avoid identical data points
        data_paths = data_paths.split(',')
        for data_path in data_paths:
            print("Prepare dataset (tarfile):")
            print(data_path)
            tf = tarfile.open(data_path, "r")
            npy_file = []
            txt_file = []
            for member in tf.getmembers():
                if member.name.endswith("npy"):
                    npy_file.append(member.name)
                elif member.name.endswith("txt"):
                    txt_file.append(member.name)
            
            txt_file = set(txt_file)
            for codec_file in npy_file:
                if codec_file in ignore_file:
                    continue
                
                array_file = BytesIO()
                array_file.write(tf.extractfile(codec_file).read())
                array_file.seek(0)
                codec_code = np.load(array_file)
                
                text = TextIOWrapper(
                        tf.extractfile(codec_file.replace(".npy", ".original.txt"))
                    ).read()
                
                if codec_file.replace(".npy", ".normalized.txt") not in txt_file:
                    text_norm = text
                else:
                    text_norm = TextIOWrapper(
                        tf.extractfile(codec_file.replace(".npy", ".normalized.txt"))
                    ).read()
                
                if len(text_norm) == 0:
                    continue
                cmu_sequence = intersperse(
                    text_to_sequence(text_norm, ["english_cleaners"], cmu_dict),
                    len(symbols)
                )
                
                length = int(float(TextIOWrapper(
                    tf.extractfile(codec_file.replace(".npy", ".len.txt"))
                ).read()))
                
                speaker_id = get_speaker_id(codec_file)
                if speaker_id not in self.speaker_dict.keys():
                    # if length * 320 / sample_rate > 3:
                    #     self.speaker_dict[speaker_id] = [codec_code[:, :length]]
                    # else:
                    #     self.speaker_dict[speaker_id] = []
                    self.speaker_dict[speaker_id] = [codec_code]
                else:
                    # if length * 320 / sample_rate > 3:
                    #     self.speaker_dict[speaker_id].append(codec_code[:, :length])
                    self.speaker_dict[speaker_id].append(codec_code)
                
                self.item_list.append(
                    {
                        "code": self.normalization(
                            torch.FloatTensor(np.array(codec_code / 1023))
                        ),
                        "ignore_code": codec_code,
                        "text": text,
                        "text_norm": text_norm,
                        "cmu_sequence": cmu_sequence,
                        "code_length": length,
                        "speaker_id": speaker_id
                    }
                )
            ignore_file.update(npy_file)            
            
    def __len__(self):
        return len(self.item_list)
    
    def __getitem__(self, idx):
        item = self.item_list[idx]
        random_sample = self.get_random_sample(item["speaker_id"], item["ignore_code"])
        
        item["sample"] = self.normalization(
            torch.FloatTensor(np.array(random_sample / 1023))
        )
        
        return item
    
    def get_random_sample(self, speaker_id, ignore_code) -> np.ndarray:
        sample_list = self.speaker_dict[speaker_id]
        sample_list = [code for code in sample_list if code.all() != ignore_code.all()]
        
        if len(sample_list) == 0:
            sample = ignore_code
        else:
            sample = random.sample(sample_list, k=1)[0]
        
        return sample
            
            
class TTS_MultiSpkr_Collate_Fn(object):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        
    def __call__(self, batch):
        batch_code = []
        batch_text = []
        batch_len = []
        batch_cmu = []
        batch_sample = []
        for item in batch:
            batch_code.append(item["code"])
            batch_text.append(item["text_norm"])
            batch_len.append(item["code_length"])
            batch_cmu.append(torch.tensor(item["cmu_sequence"]))
            batch_sample.append(torch.tensor(item["sample"]))
            
        return {
            "code": batch_code,
            "sample": batch_sample,
            "text": batch_text,
            "code_length": batch_len,
            "cmu_sequence": batch_cmu,
        }