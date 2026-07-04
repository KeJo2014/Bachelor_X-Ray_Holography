import os
import torch
import numpy as np
import logging
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets.abstract_dataset import AbstractDataset
from sklearn.model_selection import train_test_split

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

        # DUMMY LABEL DATA
        num_classes = 3
        multi_hot_label = torch.zeros(num_classes, dtype=torch.float32)
        active_classes = [0, 2]
        for c in active_classes:
            multi_hot_label[c] = 1.0

        _, h, w = tensor.shape
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0
        mask_np = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= 27**2
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        return (
            tensor,
            multi_hot_label,
            mask_tensor,
        )


class HologramDataModule(AbstractDataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = min(8, max(1, (os.cpu_count() or 1) - 1)),
        limit_samples: int = None,
    ):
        super().__init__(
            data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
        )
        self.img_size = 960
        self.setup_loaded = False
        self.limit_samples = limit_samples

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
        logger.info(f"{len(valid_files)} holograms found")

        if not valid_files:
            raise ValueError(f"No valid files found in directory: {self.data_dir}")

        if self.limit_samples is not None and self.limit_samples < len(valid_files):
            logger.info(
                f"Reducing dataset of {len(valid_files)} to {self.limit_samples} instances."
            )
            valid_files, _ = train_test_split(
                valid_files,
                train_size=self.limit_samples,
                random_state=42,
            )

        train_files, temp_files = train_test_split(
            valid_files,
            test_size=0.2,
            random_state=42,
        )
        val_files, test_files = train_test_split(
            temp_files,
            test_size=0.5,
            random_state=42,
        )

        self.train_dataset = HologramDataset(
            file_paths=train_files,
            transform=self.transform,
        )
        self.val_dataset = HologramDataset(
            file_paths=val_files,
            transform=self.transform,
        )
        self.test_dataset = HologramDataset(
            file_paths=test_files,
            transform=self.transform,
        )

        self.setup_loaded = True
