from pathlib import PurePosixPath
from typing import Any, Dict
import numpy as np
import fsspec
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path

class NumPyDataSet(AbstractDataset[np.ndarray, None]):
    """
    `NumPyDataSet` loads/saves data from/to `.npy` files as NumPy arrays.

    Example:
    ::

        >>> dataset = NumPyDataSet(filepath='/path/to/file.npy')
        >>> dataset.save(data=np.array([1, 2, 3]))
        >>> data = dataset.load()
    """

    def __init__(self, filepath: str):
        """
        Creates a new instance of `NumPyDataSet` to load/save data from/to `.npy` files.

        Args:
            filepath: The location of the `.npy` file to load/save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def load(self) -> np.ndarray:
        """
        Loads data from the `.npy` file.

        Returns:
            Data from the `.npy` file as a NumPy array.
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, 'rb') as f:
            return np.load(f, allow_pickle=True)

    def save(self, data: np.ndarray) -> None:
        """
        Saves data to the `.npy` file.

        Args:
            data: The NumPy array to save to the `.npy` file.
        """
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, 'wb') as f:
            np.save(f, data)

    def _describe(self) -> Dict[str, Any]:
        """
        Returns a dictionary describing the dataset attributes.

        Returns:
            A dictionary with the dataset attributes.
        """
        return {
            "protocol": self._protocol,
            "filepath": str(self._filepath),
            "filesystem": str(self._fs),
        }
