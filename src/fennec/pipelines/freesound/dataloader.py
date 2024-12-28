import torch
from torch.utils.data import Dataset, DataLoader
class AudioFeatureDataset(Dataset):
    def __init__(self, data,pre_processing ,train=True,pre_processing_parameters=None):
        """
        Initialize the dataset.
        :param processed_data: Dictionary with train and eval features.
        :param labels: Dictionary or list of labels for each audio sample.
        :param train: Boolean indicating whether to use train or eval features.
        """
        self.data = data
        self.labels = list(data.keys())
        self.audios = list(data.values())
        self.train = train
        self.pre_processing = pre_processing
        self.pre_processing_parameters = pre_processing_parameters or {}

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio = self.audios[idx]
        fs, audio_data = audio() 
        # Apply the preprocessing function if provided
        if self.pre_processing:
            audio_data = self.pre_processing(
                audio_data, 
                self.pre_processing_parameters.get('padding'), 
                self.pre_processing_parameters.get('mfcc')
            )
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        
        return audio_tensor, label