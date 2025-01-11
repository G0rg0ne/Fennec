import torch
from torch.utils.data import Dataset, DataLoader

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