import torch
from torch.utils.data import DataLoader, Dataset
import webdataset as wds
import tarfile
import numpy as np
from io import BytesIO, TextIOWrapper
from utils import get_cwd
from process_text import text_to_sequence, cmudict, sequence_to_text
from process_text.symbols import symbols

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


class SingleSpeakerDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        cwd = get_cwd()    
        cmu_dict = cmudict.CMUDict(cwd + "/tts/process_text/cmu_dictionary")
        
        tf = tarfile.open(data_path, "r")
        npy_file = []
        txt_file = []
        for member in tf.getmembers():
            if member.name.endswith("npy"):
                npy_file.append(member.name)
            elif member.name.endswith("txt"):
                txt_file.append(member.name)
        
        txt_file = set(txt_file)
        self.item_list = []
        for codec_file in npy_file:
            array_file = BytesIO()
            array_file.write(tf.extractfile(codec_file).read())
            array_file.seek(0)
            codec_code = np.load(array_file)
            
            text = TextIOWrapper(
                tf.extractfile(codec_file.replace(".npy", ".txt"))
            ).read()
            
            cmu_sequence = intersperse(
                text_to_sequence(text, ["english_cleaners"], cmu_dict),
                len(symbols)
            )
            
            length = float(TextIOWrapper(
                tf.extractfile(codec_file.replace(".npy", ".len.txt"))
            ).read())
            
            if codec_file.replace(".npy", ".normalized.txt") not in txt_file:
                self.item_list.append(
                    {
                        "code": codec_code,
                        "text": text,
                        "cmu_sequence": cmu_sequence,
                        "code_length": length
                    }
                )
            else:                
                text_norm = TextIOWrapper(
                    tf.extractfile(codec_file.replace(".npy", ".normalized.txt"))
                ).read()
                
                self.item_list.append(
                    {
                        "code": codec_code,
                        "text": text,
                        "text_norm": text_norm,
                        "cmu_sequence": cmu_sequence,
                        "code_length": length
                    }
                )
            
            
    def __len__(self):
        return len(self.item_list)
    
    def __getitem__(self, idx):
        return self.item_list[idx]


def tts_collation_fn(batch):
    batch_code = []
    batch_text = []
    for item in batch:
        batch_code.append(item["code"])
        batch_text.append(item["text"])
    
    if "text_norm" in batch[0].keys():
        batch_text_norm = [item["text_norm"] for item in batch]
        
        return {
            "code": torch.tensor(np.array(batch_code)).squeeze(),
            "text": batch_text,
            "text_norm": batch_text_norm
        }
    else:
        return {
            "code": torch.tensor(np.array(batch_code)).squeeze(),
            "text": batch_text
        }


def create_dataloader(data_file, batch_size):
    dataset = SingleSpeakerDataset(data_file)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=tts_collation_fn)
    
    return dataloader
