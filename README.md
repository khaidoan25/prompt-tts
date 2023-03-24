# Prompt-TTS

## Prepare dataset

We will you [WebDataset](https://github.com/webdataset/webdataset) format.

Create a tar file including wav files and transcripts.

### TTS

`tts-examples.tar`

```bash
file1.wav
file1.txt
file2.wav
file2.txt
...
```

### Prompt-TTS

`prompt-tts-examples.tar`

```bash
speaker_1/file1.wav
speaker_1/file1.txt
speaker_2/file1.wav
speaker_2/file1.txt
...
```

Run [generate_code](./data_preparation/generate_code.py) to generate codec codes matrix

```bash
python ./data_preparation/generate_code.py --input_file <tarfile> --batch_size=16
```

