import os
import torch
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class HologramDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        :param file_paths: list with absolute paths to all data files
        :param transform: optional pytorch transformation
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        holo = np.fromfile(filename, dtype=np.float64).reshape((972, 960))

        # preprocess hologram: rotation, clipping and apply logarithm
        holo = np.rot90(holo, k=1).copy()
        holo = np.clip(holo, 0, None)
        holo = np.log1p(holo)

        # normalize specific image
        h_min = holo.min()
        h_max = holo.max()
        if h_max > h_min:
            holo = (holo - h_min) / (h_max - h_min)
        else:
            holo = holo - h_min

        # convert to pytorch tensor and add channel dimension [1, H, W]
        tensor = torch.from_numpy(holo).float().unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, 0  # TODO: return real label instead of 0


class HologramDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = min(8, max(1, (os.cpu_count() or 1) - 1)),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 960
        self.setup_loaded = False

        self.transform = T.Compose(
            [T.CenterCrop((self.img_size, self.img_size))]
        )  # crop hologram to quadratic format

    def setup(self, stage=None):
        if self.setup_loaded:
            return
        all_files = [
            os.path.join(self.data_dir, f"Raw_Hologram_{i:05d}.bin")
            for i in range(0, 601)
        ]
        valid_files = [f for f in all_files if os.path.exists(f)]
        logger.info(f"{len(valid_files)} hologramms found")

        full_dataset = HologramDataset(valid_files, transform=self.transform)

        # three-way split (80% Train, 10% Val, 10% Test)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        self.setup_loaded = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
        )
