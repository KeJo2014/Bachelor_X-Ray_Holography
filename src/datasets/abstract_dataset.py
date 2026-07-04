import os
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class AbstractDataset(pl.LightningDataModule, ABC):
    """
    This is the abstract base class for datasets.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = min(8, max(1, (os.cpu_count() or 1) - 1)),
    ) -> None:
        """
        Initializes the Experiment.

        :param name: Name of the experiment.
        :return: None
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def setup(self, stage=None) -> None:
        """
        This method loads and prepares the dataset
        """
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
        )
