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
            
            if codec_file.replace(".npy", ".normalized.txt") not in txt_file:
                text_norm = text
            else:
                text_norm = TextIOWrapper(
                    tf.extractfile(codec_file.replace(".npy", ".normalized.txt"))
                ).read()
            
            cmu_sequence = intersperse(
                text_to_sequence(text_norm, ["english_cleaners"], cmu_dict),
                len(symbols)
            )
            
            length = float(TextIOWrapper(
                tf.extractfile(codec_file.replace(".npy", ".len.txt"))
            ).read())
            
            if codec_file.replace(".npy", ".normalized.txt") not in txt_file:
                self.item_list.append(
                    {
                        "code": codec_code / 1023,
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
                        "code": codec_code / 1023,
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
    batch_len = []
    batch_cmu = []
    for item in batch:
        batch_code.append(item["code"])
        batch_text.append(item["text"])
        batch_len.append(item["code_length"])
        batch_cmu.append(item["cmu_sequence"])
    
    if "text_norm" in batch[0].keys():
        batch_text_norm = [item["text_norm"] for item in batch]
        
        return {
            "code": torch.tensor(np.array(batch_code)).squeeze(),
            "text": batch_text,
            "text_norm": batch_text_norm,
            "code_length": batch_len,
            "cmu_sequence": batch_cmu
        }
    else:
        return {
            "code": torch.tensor(np.array(batch_code)).squeeze(),
            "text": batch_text,
            "code_length": batch_len,
            "cmu_sequence": batch_cmu
        }
        
        
def _collate_batch_helpler(
    examples,
    pad_token_id,
    max_length,
    return_mask=False,
):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], 0, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


class TTS_SingleSpkr_Collate_Fn(object):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        
    def __call__(self, batch):
        batch_code = []
        batch_text = []
        batch_len = []
        batch_cmu = []
        for item in batch:
            batch_code.append(item["code"])
            batch_text.append(item["text"])
            batch_len.append(item["code_length"])
            batch_cmu.append(item["cmu_sequence"])
            
        # text_seq, text_mask = _collate_batch_helpler(batch_text, )
        cmu_seq, cmu_mask = _collate_batch_helpler(
            batch_cmu, 
            0,
            self.max_seq_length,
            return_mask=True
        )
        
        if "text_norm" in batch[0].keys():
            batch_text_norm = [item["text_norm"] for item in batch]
            
            return {
                "code": torch.FloatTensor(np.array(batch_code)),
                "text": batch_text,
                "text_norm": batch_text_norm,
                "code_length": batch_len,
                "cmu_sequence": batch_cmu,
                "cmu_sequence_id": torch.IntTensor(cmu_seq),
                "attention_mask": torch.IntTensor(cmu_mask),
            }
        else:
            return {
                "code": torch.FloatTensor(np.array(batch_code)),
                "text": batch_text,
                "code_length": batch_len,
                "cmu_sequence": batch_cmu,
                "cmu_sequence_id": torch.IntTensor(cmu_seq),
                "attention_mask": torch.IntTensor(cmu_mask),
            }
            

def create_dataloader(data_file, batch_size, max_seq_length):
    dataset = SingleSpeakerDataset(data_file)
    
    my_collate = TTS_SingleSpkr_Collate_Fn(max_seq_length)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    
    return dataloader
