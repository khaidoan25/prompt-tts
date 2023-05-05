# Prompt-TTS

## Environment

We're using Python 3.8 for this repo. After installing a virtual environment
(use `virtualenv` or `conda`), we install `poetry` with pip.

Then run `poetry install` to install the requirements.

## Prepare dataset

### TTS

We will use LJSpeech dataset. Link to download is [hear](https://keithito.com/LJ-Speech-Dataset/)

Then extract the transcriptions

```bash
python ./data_preparation/extract_transcript.py --input_file <path to ljspeech data>
```

Then we need to generate the codec code from wav files.

Run [generate_code](./data_preparation/generate_code.py) to generate codec codes matrix

```bash
python ./data_preparation/generate_code.py --input_file <tarfile>
```

Then we will have a data file with the name `LJSpeech-1.1_processed.tar`

## Training

Then run the training script

```bash
accelerate launch train.py \
    --data_file <path_to_data_file> \
    --log_dir <path_to_log_directory> \
    --config_file <run_code/1d_config> \
    --ckpt_dir <path_to_checkpoint_directory>
```

We can use tensorboard to monitor the training loss

```bash
tensorboard --logidr=<path_to_log_directory>
```
