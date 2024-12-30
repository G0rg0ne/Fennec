"""
This is a boilerplate pipeline 'freesound'
generated using Kedro 0.19.10
"""

from .utils import (
    stft,
    mel_filter,
    signal_power_to_db,
    discrete_cos_transformation,
    sin_liftering,
)
import numpy as np
import librosa
from pathlib import Path
from typing import Any, Callable
import torch
from .dataloader import AudioFeatureDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from torch import nn
from .model import CNNAudioClassifier
from loguru import logger
import mlflow
from tqdm import tqdm
import mlflow
import mlflow.pytorch


def training_pipeline(
    train_audio,
    eval_audio,
    train_gt,
    eval_gt,
    vocab,
    pre_processing_parameters,
    training_parameters,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    list_of_labels = vocab[1].tolist()
    le = LabelEncoder()
    class_label_ecod = le.fit(list_of_labels)

    train_gt["fname"] = train_gt["fname"].astype(str)
    train_gt_fname_to_labels = dict(zip(train_gt["fname"], train_gt["labels"]))
    eval_gt_fname_to_labels = dict(zip(eval_gt["fname"], eval_gt["labels"]))

    train_dataset = AudioFeatureDataset(
        train_audio,
        train_gt_fname_to_labels,
        preprocess_raw_data,
        train=True,
        pre_processing_parameters=pre_processing_parameters,
    )
    eval_dataset = AudioFeatureDataset(
        eval_audio,
        eval_gt_fname_to_labels,
        preprocess_raw_data,
        train=False,
        pre_processing_parameters=pre_processing_parameters,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_parameters["batch_size"],
        shuffle=True,
        num_workers=training_parameters["num_workers"],
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_parameters["batch_size"],
        shuffle=False,
        num_workers=training_parameters["num_workers"],
    )
    input_size = tuple(training_parameters["input_size"])
    num_classes = training_parameters["num_classes"]  # Number of target classes
    learning_rate = training_parameters[
        "learning_rate"
    ]  # Learning rate for the optimizer
    num_epochs = training_parameters["num_epochs"]

    model = CNNAudioClassifier(input_size=input_size, num_classes=num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", training_parameters["batch_size"])
        mlflow.log_param("num_epochs", num_epochs)

        for epoch in range(num_epochs):
            epoch_loss = 0
            model.train()
            with tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                unit="batch",
            ) as pbar:
                for idx, data in enumerate(train_loader):
                    # Transform class labels
                    data[1] = class_label_ecod.transform(data[1])
                    data[1] = torch.LongTensor(data[1])

                    # Move inputs and labels to the device
                    inputs, labels = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion(outputs, labels)
                    epoch_loss += loss.item()

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Update progress bar
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

            # Log metrics for the epoch
            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}"
            )

            # Evaluation Loop
            model.eval()
            eval_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in eval_loader:
                    # Transform class labels
                    data[1] = class_label_ecod.transform(data[1])
                    data[1] = torch.LongTensor(data[1])

                    # Move inputs and labels to the device
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item()

                    # Compute accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_eval_loss = eval_loss / len(eval_loader)
            accuracy = correct / total
            mlflow.log_metric("avg_eval_loss", avg_eval_loss, step=epoch)
            mlflow.log_metric("eval_accuracy", accuracy, step=epoch)
            logger.info(
                f"Validation Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

        # Log the final model
        mlflow.pytorch.log_model(model, "model")
        logger.info("Model logged to MLflow.")
    return model


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
    mfcc_features = mfcc_extraction(
        data_set_fix_length,
        mfcc_parameters["fs"],
        mfcc_parameters["n_fft"],
        mfcc_parameters["frame_size"],
        mfcc_parameters["frame_step"],
        mfcc_parameters["n_mels"],
        mfcc_parameters["n_mfcc"],
    )

    return mfcc_features


def pad_sample(sample: tuple, pad_params: dict):
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
    padded_audio = librosa.util.fix_length(
        audio_array, size=pad_params["fix_length"], axis=0, mode=pad_params["mode"]
    )

    return padded_audio


def mfcc_extraction(
    data_set_fix_length: list,
    fs: float,
    n_fft: int,
    frame_size: float,
    frame_step: float,
    n_mels: int,
    n_mfcc: int,
):
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
    mfcc_features = extract_mfcc_feature(
        y=data_set_fix_length,
        fs=fs,
        n_fft=n_fft,
        frame_size=frame_size,
        frame_step=frame_step,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
    )
    return mfcc_features


def extract_mfcc_feature(
    y: np.array,
    fs: float,
    n_fft: int = 512,
    frame_size: float = 0.025,
    frame_step: float = 0.01,
    n_mels: int = 40,
    n_mfcc: int = 13,
):
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
    mel_power_spec_frames, hz_freq = mel_filter(pow_spec_frames, 0, fs / 2, n_mels, fs)

    # log of signal
    log_spec_frames = signal_power_to_db(mel_power_spec_frames)

    # perform DCT
    mfcc = discrete_cos_transformation(log_spec_frames)

    # liftering of mfcc
    mfcc = sin_liftering(mfcc)

    # take just lwo frequencies mfcc
    mfcc = mfcc[:, 1:n_mfcc]

    return mfcc
