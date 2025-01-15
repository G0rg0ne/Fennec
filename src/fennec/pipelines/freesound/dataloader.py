import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class AudioFeatureDataset(Dataset):
    def __init__(self, data, pre_processing, train=True, pre_processing_parameters=None):
        """
        Initialize the dataset.
        :param data: Dictionary with audio data.
        :param fname_to_labels: Dictionary mapping filenames to labels as comma-separated strings.
        :param pre_processing: Function to preprocess audio data.
        :param train: Boolean indicating whether to use train or eval features.
        :param pre_processing_parameters: Optional preprocessing parameters.
        :param label_encoder: LabelEncoder to encode labels.
        """
        self.data = data
        self.fname = list(data.keys())
        self.audios = list(data.values())
        self.train = train
        self.pre_processing = pre_processing
        self.pre_processing_parameters = pre_processing_parameters or {}

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        fname = self.fname[idx]
        audio = self.audios[idx]
        
        # Load audio data
        fs, audio_data = audio()
        
        # Apply preprocessing function if provided
        if self.pre_processing:
            audio_data = self.pre_processing(
                audio_data,
                self.pre_processing_parameters.get('padding'),
                self.pre_processing_parameters.get('mfcc')
            )

        return fname,audio_data