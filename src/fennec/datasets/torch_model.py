from pathlib import PurePosixPath
from typing import Any, Dict

import torch
import fsspec
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class TorchModelDataset(AbstractDataset[torch.nn.Module, torch.nn.Module]):
    """``TorchModelDataset`` loads / saves a PyTorch model from a given filepath.

    Example:
    ::

        >>> TorchModelDataset(filepath='/models/model.pth')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of TorchModelDataset to load / save PyTorch models.

        Args:
            filepath: The location of the model file to load / save.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def load(self) -> torch.nn.Module:
        """Loads a PyTorch model from the file, automatically selecting GPU or CPU.

        Returns:
            The loaded PyTorch model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, "rb") as f:
            model = torch.load(f, map_location=device)
        return model

    def save(self, model: torch.nn.Module) -> None:
        """Saves a PyTorch model to the file.

        Args:
            model: The PyTorch model to save.
        """
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, "wb") as f:
            torch.save(model, f)

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
