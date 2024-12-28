"""
This is a boilerplate pipeline 'freesound'
generated using Kedro 0.19.10
"""
from .utils import stft, mel_filter, signal_power_to_db, discrete_cos_transformation, sin_liftering
import numpy as np
import librosa
from pathlib import Path
from typing import Any, Callable
import torch
from .dataloader import AudioFeatureDataset
from torch.utils.data import DataLoader


def pre_processing(train_audio, eval_audio, pre_processing_parameters):
    train_dataset = AudioFeatureDataset(train_audio,preprocess_raw_data,train=True,pre_processing_parameters=pre_processing_parameters)
    eval_dataset = AudioFeatureDataset(eval_audio,preprocess_raw_data,train=False,pre_processing_parameters=pre_processing_parameters)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=False)

    return None, None


def preprocess_raw_data(data: list, pad_params: dict, mfcc_parameters: dict):
    """
    Preprocess audio data. Returns MFCC features of each audio sample.

    :param data: List of audio data
    :param fix_length: Fix length to pad each audio sample
    :param mfcc_parameters: Parameters for MFCC extraction
    :return: MFCC data for each audio sample
    """

    # pad data to fix length
    data_set_fix_length = pad_sample(data, pad_params)
    # extract MFCC for each sample
    mfcc_features = mfcc_extraction(data_set_fix_length, mfcc_parameters['fs'], mfcc_parameters['n_fft'],
                                    mfcc_parameters['frame_size'], mfcc_parameters['frame_step'], mfcc_parameters['n_mels'],mfcc_parameters['n_mfcc'])

    return mfcc_features


def pad_sample(sample: tuple,pad_params: dict):
    """
    Pad a single sample (frequency, audio) to a fixed length.

    :param sample: Input sample as a tuple (frequency, audio)
    :param fix_length: Fixed length to pad the audio sample
    :return: Tuple containing the frequency and padded audio sample
    """
    audio = sample  # Unpack the tuple
    
    # Ensure the audio part is a NumPy array for compatibility with librosa
    audio_array = np.array(audio)
    # Pad the audio sample to the specified length
    padded_audio = librosa.util.fix_length(audio_array, size=pad_params['fix_length'], axis=0, mode=pad_params['mode'])
    
    return  padded_audio

def mfcc_extraction(data_set_fix_length: list, fs: float, n_fft: int, frame_size: float, frame_step: float, n_mels: int, n_mfcc: int):
    """
    Extract MFCC features for each data sample.

    :param data_set_fix_length: Input data set
    :param fs: Sampling frequency
    :param n_fft: Num of Nfft points
    :param frame_size: Size of frame in sec
    :param frame_step: Frame step in sec
    :return: MFCC features for each data sample
    """
    # Extract MFCC features for the single data sample
    mfcc_features = extract_mfcc_feature(y=data_set_fix_length, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step,n_mels=n_mels,n_mfcc= n_mfcc)
    return mfcc_features

def extract_mfcc_feature(y: np.array, fs: float, n_fft: int = 512, frame_size: float = 0.025, frame_step: float = 0.01,
                         n_mels: int = 40, n_mfcc: int = 13):
    """
    Extracts  MFCCs from input signal y.

    :param y: Input signal
    :param fs: Sampling frequency
    :param n_fft: Num of points to calculate FFT
    :param frame_size: Size of each frame in seconds
    :param frame_step: Steps between frames in seconds
    :param n_mels: Number of Mel filters
    :param n_mfcc: Number of MFCC features
    :return: MFCC features of input signal
    """

    # mag calculate spectrogram of signal
    mag_spec_frames = np.abs(stft(y, fs, n_fft, frame_size, frame_step))

    # create power spectrogram
    pow_spec_frames = (mag_spec_frames**2) / mag_spec_frames.shape[1]

    # filter signal with Mel-filter
    mel_power_spec_frames, hz_freq = mel_filter(pow_spec_frames, 0, fs/2, n_mels, fs)

    # log of signal
    log_spec_frames = signal_power_to_db(mel_power_spec_frames)

    # perform DCT
    mfcc = discrete_cos_transformation(log_spec_frames)

    # liftering of mfcc
    mfcc = sin_liftering(mfcc)

    # take just lwo frequencies mfcc
    mfcc = mfcc[:, 1:n_mfcc]

    return mfcc