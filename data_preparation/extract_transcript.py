import argparse
import tarfile
import pandas as pd
from io import BytesIO
from tqdm import tqdm


def main(file_path):
    with tarfile.open(file_path, "r:bz2") as tf:
        metadata = pd.read_csv(tf.extractfile(tf.getmember("LJSpeech-1.1/metadata.csv")), delimiter='|', header=None)
        
        
    with tarfile.open(file_path.replace(".bz2", ""), "w") as tf:
        for row in tqdm(metadata.iterrows()):
            file_name = row[1][0]
            text = row[1][1]
            text_norm = row[1][2]
            
            if isinstance(text, float):
                text = text_norm
            if isinstance(text_norm, float):
                text_norm = text
            
            string = BytesIO(bytes(text, "utf-8"))
            tarinfo = tarfile.TarInfo(f"{file_name}.txt")
            tarinfo.size = len(text)
            tf.addfile(tarinfo, string)
            
            string_norm = BytesIO(bytes(text_norm, "utf-8"))
            tarinfo_norm = tarfile.TarInfo(f"{file_name}.normalized.txt")
            tarinfo_norm.size = len(text_norm)
            tf.addfile(tarinfo_norm, string_norm)
            
    orig_tf = tarfile.open(file_path, "r:bz2")
    new_tf = tarfile.open(file_path.replace(".bz2", ""), "a")
    for member in orig_tf.getmembers():
        if member.name.endswith(".wav"):
            new_tf.addfile(member, orig_tf.extractfile(member))
    orig_tf.close()
    new_tf.close()
    

def parse_args():
    parser = argparse.ArgumentParser(description="Extract transcripts.")
    parser.add_argument('--input_file', type=str,
                        help="Path to the input file.", required=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_file)