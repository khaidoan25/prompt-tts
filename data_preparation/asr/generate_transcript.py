import torch
from torch.utils.data import DataLoader
import webdataset as wds
import argparse
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from tqdm import tqdm
import tempfile
import tarfile
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Speech2TextForConditionalGeneration.from_pretrained(
    "/home/khaidoan/projects/def-mageed/khaidoan/speech/work_dir/s2t-small-librispeech-asr"
)
processor = Speech2TextProcessor.from_pretrained(
    "/home/khaidoan/projects/def-mageed/khaidoan/speech/work_dir/s2t-small-librispeech-asr"
)
model.to(device)

with open("ignore_list.txt", "a+") as f:
    ignore_list = set(f.read().split('\n'))
    
    
def collate_fn(batch):
    def padding(data: torch.tensor, max_length: int):
        padd = torch.tensor([0]*(max_length - data.shape[-1]))
        data = torch.cat((data.squeeze(), padd), -1)
            
        return data
    
    max_length = max(item["wav"][0].shape[-1] for item in batch)
    batch = [(item["__key__"], padding(item["wav"][0], max_length), item["__url__"]) for item in batch if item["__key__"] not in ignore_list]
    
    data = [item[1].numpy() for item in batch]
    
    return {
        'path': [item[0] for item in batch],
        'data': data,
        'tar_file': [item[2] for item in batch]
    }
    

def generate(input_dir,
             batch_size,):
    dataset = wds.WebDataset(input_dir)
    dataset = dataset.decode(wds.torch_audio)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    for batch in tqdm(dataloader):
        try:
            inputs = processor(batch['data'], sampling_rate=16_000, return_tensors="pt")
            generated_ids = model.generate(input_ids=inputs["input_features"].to(device), attention_mask=inputs["attention_mask"].to(device))
            
            transcriptions = processor.batch_decode(generated_ids)
            for path, tar_file, transcript in zip(batch['path'], batch['tar_file'], transcriptions):
                transcript_file = path + ".txt"
                with tempfile.TemporaryDirectory() as dirname:
                    with open(f"{dirname}/{transcript_file}", "w") as f:
                        f.write(transcript)
                    with tarfile.open(tar_file, "a") as tf:
                        tf.add(f"{dirname}/{transcript_file}", arcname=transcript_file)
                        
                with open("ignore_list.txt", "a") as f:
                    f.write(path + '\n')
        except:
            with open("error_list.txt", "a") as f:
                for path in batch['path']:
                    f.write(path + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Generate transcripts for audios")
    parser.add_argument('--input_dir', type=str, default=None,
                        help="Path to the input directory", required=True)
    parser.add_argument('--batch_size', type=str, default=2,
                        help="Batch size for each inference")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generate(args.input_dir, int(args.batch_size))