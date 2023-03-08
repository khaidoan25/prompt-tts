# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pathlib
import soundfile as sf
import numpy as np
import json
import multiprocessing
from multiprocessing import current_process
import argparse
import tqdm
import random
import tarfile
import os
import tempfile


def save(seq, fname, index, dataset_path):
    output = np.hstack(seq)
    file_name = fname.parent / (fname.stem + f"_{index:04}")
    file_name = f"{str(file_name).replace('/', '_')}.wav"
    
    p = current_process()
    
    with tempfile.TemporaryDirectory() as dirname:
        sf.write(f"{dirname}/{file_name}", output, samplerate=16000)
    
        with tarfile.open(dataset_path.replace(".tar", f"_{p._identity[0]}.tar"), "a") as tf:
            tf.add(f"{dirname}/{file_name}", arcname=file_name)


def cut_sequence(data, samplerate, vad, path_out, dataset_path):
    assert len(data.shape) == 1
    assert samplerate == 16000

    to_stitch = []
    length_accumulated = 0.0

    i = 0
    target_len_sec = random.choices([10, 20])[0]
    for start, end in vad:
        start_index = int(start * samplerate)
        end_index = int(end * samplerate)
        slice = data[start_index:end_index]

        # if a slice is longer than target_len_sec, we put it entirely in it's own piece
        if length_accumulated + (end - start) > target_len_sec and length_accumulated > 0:
            save(to_stitch, path_out, i, dataset_path)
            to_stitch = []
            i += 1
            length_accumulated = 0
            target_len_sec = random.choices([10, 20])[0]

        to_stitch.append(slice)
        length_accumulated += end - start

    if to_stitch:
        save(to_stitch, path_out, i, dataset_path)


def cut_book(task):
    
    path_book, dataset_path, ignore_list = task

    speaker = pathlib.Path(path_book.parent.name)

    for i, meta_file_path in enumerate(path_book.glob('*.json')):
        try:
            sound_file = meta_file_path.parent / (meta_file_path.stem + '.flac')
            
            if str(sound_file) in ignore_list:
                continue
            
            with open(meta_file_path, 'r') as f:
                meta = json.loads(f.read())
            book_id = meta['book_meta']['id']
            vad = meta['voice_activity']

            path_out = speaker / book_id / (meta_file_path.stem)
            cut_sequence(sound_file, vad, path_out, dataset_path)
            
            with open("ignore_list.txt", "a") as f:
                f.write(str(sound_file) + "\n")
        except:
            with open("error_list.txt", "a") as f:
                f.write(str(sound_file) + "\n")
                
                
def cut_book_tarfile(task):    
    meta_file, dataset_path, tar_file = task    
    meta_file_path = pathlib.Path(meta_file.name)
    sound_file_path = meta_file.name.replace('.json', '.flac')
    speaker = pathlib.Path(pathlib.Path(sound_file_path).parent.parent.name)
    
    try:
        with tarfile.open(tar_file, "r") as tf:
            # Get TarObject before extracting file
            sound_file, samplerate = sf.read(tf.extractfile(tf.getmember(sound_file_path)))
            meta = json.loads(tf.extractfile(meta_file).read())
            
        sound_file_path = sound_file_path.replace('/', '_')
        book_id = meta['book_meta']['id']
        vad = meta['voice_activity']
        
        path_out = speaker / book_id / (meta_file_path.stem)
        cut_sequence(sound_file, samplerate, vad, path_out, dataset_path)
    
        with open("ignore_list.txt", "a") as f:
            f.write(str(meta_file_path) + "\n")
    except:
        with open("error_list.txt", "a") as f:
            f.write(str(meta_file_path) + "\n")
        
                
def process_directory(directory, ignore_list):
    list_dir = pathlib.Path(directory).glob('*/*')
    list_dir = [x for x in list_dir if x.is_dir()]

    print(f"{len(list_dir)} directories detected")
    
    tasks = [
        (path_book, f"{directory + '.json'}", ignore_list)
        for path_book in list_dir
    ]
    
    return tasks

                
def process_tarfile(tar_file, ignore_list):
    tasks = None
    with tarfile.open(tar_file, "r") as tf:
        members = [i for i in tf.getmembers() if i.name.endswith('.json') and i.name not in ignore_list]    
    
    print(f"Processing subset {tar_file}")
    print(f"{len(members)} audiobooks detected")
    
    tasks = [
        (meta_file, f"{tar_file[:-4] + '_vad.tar'}", tar_file)
        for meta_file in members
    ]
    
    return tasks


def cut(input_dir,
        n_process=32,):

    with open("ignore_list.txt", "a+") as f:
        ignore_list = {row.strip() for row in f}
    with open("error_list.txt", "a+") as f:
        error_list = {row.strip() for row in f}
        
    ignore_list.update(error_list)
    
    print(f"Processing subset {input_dir}")
    print(f"Launching {n_process} processes")    
    if input_dir.endswith('tar'):
        tasks = process_tarfile(input_dir, ignore_list)
        with multiprocessing.Pool(processes=n_process) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(cut_book_tarfile, tasks), total=len(tasks)):
                pass
    else:
        tasks = process_directory(input_dir, ignore_list)
        with multiprocessing.Pool(processes=n_process) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(cut_book, tasks), total=len(tasks)):
                pass


def parse_args():

    parser = argparse.ArgumentParser(description="Cut a dataset in small "
                                     "sequences using VAD files")
    parser.add_argument('--input_dir', type=str, default=None,
                        help="Path to the input directory", required=True)

    parser.add_argument('--n_workers', type=int, default=32,
                        help="Number of parallel worker processes")


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cut(args.input_dir, args.n_workers)
