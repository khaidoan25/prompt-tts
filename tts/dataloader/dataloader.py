from torch.utils.data import DataLoader
from tts.dataloader.multi import MultiSpeakerDataset, TTS_MultiSpkr_Collate_Fn
from tts.dataloader.single import SingleSpeakerDataset, TTS_SingleSpkr_Collate_Fn
            

def create_dataloader(
    data_file,
    batch_size,
    max_seq_length,
    data_type,
    shuffle=False):
    
    if data_type == "single_speaker":
        dataset = SingleSpeakerDataset(data_file)
        my_collate = TTS_SingleSpkr_Collate_Fn(max_seq_length)
    elif data_type == "multi_speaker":
        dataset = MultiSpeakerDataset(data_file)
        my_collate = TTS_MultiSpkr_Collate_Fn(max_seq_length)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate)
    
    return dataloader
