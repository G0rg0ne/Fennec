"""
This is a boilerplate pipeline 'mfcc'
generated using Kedro 0.19.10
"""
import numpy as np
import scipy.io.wavfile as sci_wav
from matplotlib import pyplot as plt
from .utils import stft, signal_power_to_db
import librosa
import librosa.display as libdisplay

def monitoring_signal(input):
    fig = None
    temporal_audio_figs = {}
    librosa_audio_figs = {}
    for id, audio in input.items():
        fs, y = audio()
        y = 1.0 * y
        t = np.linspace(0, y.shape[0] / fs, y.shape[0])
        temporal_audio_figs[id] = plot_audio_signal(t, y, fs)
        # set parameters
        n_fft = 512
        params = {
            'fs': fs,
            'n_fft': n_fft,
            'window_length': n_fft,
            'window_step': int(n_fft/3),
        }

        # show spectrogram
        y_spec_lib, y_spec = get_spectrogram_of_signal(y, params, fs)
        #visualize_spectrogram(y_spec, params, "Implemented Spectrogram function")
        librosa_audio_figs[id] = visualize_spectrogram(y_spec_lib, params, "Librosa Spectrogram function")
    return temporal_audio_figs,librosa_audio_figs

def visualize_spectrogram(y_spec: np.array, parameters: dict, title: str = ""):
    figure = plt.figure(figsize=(8, 6))
    plt.title(title)
    libdisplay.specshow(y_spec, y_axis='linear', sr=parameters['fs'], cmap='autumn', x_axis='time', hop_length=parameters['window_step'])
    plt.show(block=False)
    return figure

def get_spectrogram_of_signal(signal: np.array, parameters: dict,fs: int):
    """
    Get Spectrogram form librosa and algorithm implementation.

    :param signal: input signal
    :param parameters: Params used for Spectrogram extraction
    :return: Spectrogram from librosa and implementation
    """

    n_fft = parameters['n_fft']
    window_length = parameters['window_length']
    window_step = parameters['window_step']

    # Use librosa for spectrogram extraction
    signal_spec_lib = librosa.amplitude_to_db(np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=window_step,
                                                                  win_length=window_length)))

    frame_size = window_length / fs
    frame_step = window_step / fs
    # Use implemented function for spectrogram extraction
    power_spectrum_frames = np.abs(stft(y=signal, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step))**2
    signal_spec = signal_power_to_db(power_spectrum_frames)

    return signal_spec_lib, signal_spec.T

def plot_audio_signal(t, y, fs):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(t, y)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Signal')
    plt.show()
    return fig