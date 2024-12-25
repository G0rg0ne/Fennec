from pathlib import PurePosixPath
from typing import Any, Dict

import scipy.io.wavfile as sci_wav
import fsspec
import numpy as np

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path
class WaveDataSet(AbstractDataset[np.ndarray, np.ndarray]):
    """``ImageDataset`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Example:
    ::

        >>> ImageDataset(filepath='/img/file/path.png')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataset to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        fs, y = sci_wav.read(load_path)
        return fs, y
    def save(self, data: np.ndarray, sample_rate: int) -> None:
        """Saves data to the wave file.

        Args:
            data: The data to save to the wave file as a numpy array.
            sample_rate: The sample rate of the wave file.
        """
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, 'wb') as f:
            sci_wav.write(f, sample_rate, data)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.

        Returns:
            A dictionary with the dataset attributes.
        """
        return {
            "protocol": self._protocol,
            "filepath": str(self._filepath),
            "filesystem": str(self._fs),
        }