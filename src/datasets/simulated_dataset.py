import os
import torch
import numpy as np
import logging
import h5py
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from datasets.abstract_dataset import AbstractDataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CDICropAndBinTransform:
    """
    Computes central crop with average pixel binning for CDI data
    Use average for diffration image and max pooling for the beamstop mask
    """

    def __init__(self, crop_size: int, target_size: int):
        self.crop_size = crop_size
        self.target_size = (target_size, target_size)

    def __call__(self, image, mask):
        image = TF.center_crop(image, output_size=[self.crop_size, self.crop_size])
        mask = TF.center_crop(mask, output_size=[self.crop_size, self.crop_size])

        image = F.adaptive_avg_pool2d(image, self.target_size)
        mask = F.adaptive_max_pool2d(mask, self.target_size)

        return image, mask


class HologramDataset(Dataset):
    def __init__(
        self,
        h5_filepath,
        run_keys,
        label_map,
        transform=None,
        global_mean=0.0,
        global_std=1.0,
    ):
        """
        :param h5_filepath: path to the h5 data file
        :param run_keys: list of specific h5 run keys assigned to this split
        :param label_map: dict mapping labels to class integers
        :param transform: pytorch transformation
        """
        self.h5_filepath = h5_filepath
        self.run_keys = run_keys
        self.label_map = label_map
        self.transform = transform
        self.global_mean = global_mean
        self.global_std = global_std
        self.h5_file = None

    def open_hdf5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_filepath, "r")

    def __len__(self):
        return len(self.run_keys) * 2  # each run yields two holograms (CL and CR)

    def __getitem__(self, idx):
        self.open_hdf5()

        run_idx = idx // 2
        is_cr = idx % 2
        run_key = self.run_keys[run_idx]
        run_data = self.h5_file[run_key]

        if is_cr == 0:
            holo = run_data["CL"]["detected"][:]
        else:
            holo = run_data["CR"]["detected"][:]

        mask_np = run_data["beamstop_mask"][:]
        holo = np.squeeze(holo)
        mask_np = np.squeeze(mask_np)

        label_val = run_data["metadata"]["magnetic_pattern"]["pattern_type_method"][()]
        if isinstance(label_val, bytes):
            label_val = label_val.decode("utf-8")

        # Preprocess hologram
        holo = np.array(holo, dtype=np.float32)
        holo = np.rot90(holo, k=1).copy()
        holo = np.clip(holo, 0, None)
        holo = np.log1p(holo)

        holo = (holo - self.global_mean) / self.global_std

        # convert to pytorch tensor with dim [1, H, W]
        tensor = torch.from_numpy(holo).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        # if applicable apply transformation
        if self.transform:
            tensor, mask_tensor = self.transform(tensor, mask_tensor)

        # generate oen-hot-tensor map
        num_classes = len(self.label_map)
        label_tensor = torch.zeros(num_classes, dtype=torch.float32)
        label_tensor[self.label_map[label_val]] = 1.0

        return tensor, label_tensor, mask_tensor


class HologramDataModule(AbstractDataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = min(8, max(1, (os.cpu_count() or 1) - 3)),
    ):
        super().__init__(
            data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
        )
        self.img_size = 960
        self.setup_loaded = False
        self.initial_crop_size = 1280
        self.global_mean = 0.0
        self.global_std = 1.0

        self.transform = CDICropAndBinTransform(
            crop_size=self.initial_crop_size, target_size=self.img_size
        )

    def _compute_global_stats(self, data_file, train_keys):
        """
        Calculate iteratively global mean and std for training set.
        """
        logger.info("Computing global mean and std over training set...")
        sum_val = 0.0
        sq_sum_val = 0.0
        pixel_count = 0

        for key in train_keys:
            for pol in ["CL", "CR"]:
                holo = data_file[key][pol]["detected"][:]
                holo = np.squeeze(holo)

                holo = np.array(holo, dtype=np.float64)
                holo = np.clip(holo, 0, None)
                holo = np.log1p(holo)

                sum_val += holo.sum()
                sq_sum_val += (holo**2).sum()
                pixel_count += holo.size

        mean = sum_val / pixel_count
        variance = (sq_sum_val / pixel_count) - (mean**2)
        std = np.sqrt(max(variance, 1e-8))

        logger.info(f"Global stats computed - Mean: {mean:.4f}, Std: {std:.4f}")
        return float(mean), float(std)

    def setup(self, stage=None):
        if self.setup_loaded:
            return

        # open file to discover keys and classes
        with h5py.File(self.data_dir, "r") as f:
            data = f
            run_keys = [k for k in f.keys() if k != "_pipeline_config"]

            # find all unique classes
            all_labels = set()
            run_labels = []
            for key in run_keys:
                lbl = data[key]["metadata"]["magnetic_pattern"]["pattern_type_method"][
                    ()
                ]
                if isinstance(lbl, bytes):
                    lbl = lbl.decode("utf-8")
                all_labels.add(lbl)
                run_labels.append(lbl)

                # check general format
                assert "CL" in data[key].keys()
                assert "CR" in data[key].keys()

            # create map of all unique labels
            unique_labels = sorted(list(all_labels))
            label_map = {lbl: i for i, lbl in enumerate(unique_labels)}

            logger.info(
                f"Found {len(run_keys)*2} total samples from {len(run_keys)} runs."
            )
            logger.info(f"Detected classes: {label_map}")

            # stratified group shuffle split with 80% train, 10% validation and 10% test split
            # ensuring CL and CR from the same run stick together -> prevent data leakage
            train_keys, temp_keys, train_labels, temp_labels = train_test_split(
                run_keys,
                run_labels,
                test_size=0.2,
                stratify=run_labels,
                random_state=42,
            )

            # Zweiter Split: Temp hälftig in Val (10%) und Test (10%) aufteilen
            val_keys, test_keys, _, _ = train_test_split(
                temp_keys,
                temp_labels,
                test_size=0.5,
                stratify=temp_labels,
                random_state=42,
            )
            self.global_mean, self.global_std = self._compute_global_stats(
                f, train_keys
            )

        # instantiate datasets
        self.train_dataset = HologramDataset(
            self.data_dir,
            train_keys,
            label_map,
            transform=self.transform,
            global_mean=self.global_mean,
            global_std=self.global_std,
        )
        self.val_dataset = HologramDataset(
            self.data_dir,
            val_keys,
            label_map,
            transform=self.transform,
            global_mean=self.global_mean,
            global_std=self.global_std,
        )
        self.test_dataset = HologramDataset(
            self.data_dir,
            test_keys,
            label_map,
            transform=self.transform,
            global_mean=self.global_mean,
            global_std=self.global_std,
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
