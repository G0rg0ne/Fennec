from pathlib import PurePosixPath
from typing import Any, Dict
import torch
import fsspec
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class TorchTensorDataSet(AbstractDataset[torch.Tensor, None]):
    """
    `TorchTensorDataSet` loads/saves data from/to files as PyTorch tensors.

    Example:
    ::

        >>> dataset = TorchTensorDataSet(filepath='/path/to/file.pt')
        >>> dataset.save(data=torch.tensor([1, 2, 3]))
        >>> data = dataset.load()
    """

    def __init__(self, filepath: str):
        """
        Creates a new instance of `TorchTensorDataSet` to load/save data from/to files.

        Args:
            filepath: The location of the file to load/save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def load(self) -> torch.Tensor:
        """
        Loads data from the file as a PyTorch tensor.

        Returns:
            Data from the file as a PyTorch tensor.
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, 'rb') as f:
            return torch.load(f)

    def save(self, data: torch.Tensor) -> None:
        """
        Saves data to the file as a PyTorch tensor.

        Args:
            data: The PyTorch tensor to save to the file.
        """
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, 'wb') as f:
            torch.save(data, f)

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