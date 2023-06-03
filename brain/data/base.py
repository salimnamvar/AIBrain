"""Base Data Loader

    This file defines a base data dataset class.
"""


# region Imported Dependencies
from abc import abstractmethod
from typing import Tuple
import torch
from torch.utils.data import Dataset
# endregion Imported Dependencies


class BaseDataset(Dataset):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
