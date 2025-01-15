import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class AudioFeatureDataset(Dataset):
    def __init__(self, data, fname_to_labels, pre_processing, train=True, pre_processing_parameters=None, label_encoder=None):
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
        self.fname_to_labels = fname_to_labels
        self.fname = list(data.keys())
        self.audios = list(data.values())
        self.train = train
        self.pre_processing = pre_processing
        self.pre_processing_parameters = pre_processing_parameters or {}
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        fname = self.fname[idx]
        audio = self.audios[idx]
        labels = self.fname_to_labels[fname].split(',')  # Split comma-separated labels
        
        # Encode each label and convert to binary vector
        encoded_labels = self.label_encoder.transform(labels)
        label_vector = torch.zeros(len(self.label_encoder.classes_), dtype=torch.float32)
        label_vector[encoded_labels] = 1.0  # Set positions corresponding to labels to 1
        
        # Load audio data
        fs, audio_data = audio()

        # Apply preprocessing function if provided
        if self.pre_processing:
            audio_data = self.pre_processing(
                audio_data,
                self.pre_processing_parameters.get('padding'),
                self.pre_processing_parameters.get('mfcc')
            )
        
        # Convert audio to tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        return audio_tensor, label_vector
    
class CustomAudioDataset(Dataset):
    def __init__(self, data, labels_dict, label_encoder, num_classes):
        """
        Args:
            data (dict): Dictionary where keys are filenames and values are functions to load audio.
            labels_dict (dict): Dictionary mapping filenames to their labels.
            label_encoder (LabelEncoder): Fitted LabelEncoder for encoding labels.
            num_classes (int): Number of unique classes.
        """
        self.data = data
        self.labels_dict = labels_dict
        self.label_encoder = label_encoder
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the filename and corresponding loader function
        filename = list(self.data.keys())[idx]
        audio_loader = self.data[filename]

        # Load the audio data lazily
        audio_tensor = torch.tensor(audio_loader(), dtype=torch.float32)

        # Get the labels for the audio file
        label = self.labels_dict[filename]
        encoded_label = self.label_encoder.transform(label.split(','))

        # Create one-hot encoded tensor for the labels
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float32).scatter_(
            0, torch.tensor(encoded_label, dtype=torch.long), 1.0
        )

        return audio_tensor, label_tensor