import os
import glob
import torch
import numpy as np
import pandas as pd
import logging
import h5py
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import scipy.ndimage
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CDICropAndBinTransform:
    """
    Computes central crop with average pixel binning for CDI data
    Use average for diffraction image and max pooling for the beamstop mask
    """

    def __init__(self, crop_size: int, target_size: int, center_holograms: bool = True):
        self.crop_size = crop_size
        self.target_size = (target_size, target_size)
        self.center_holograms = center_holograms

    def __call__(self, image, mask):
        if self.center_holograms:
            image_np = image.numpy()[0]
            # apply threshold to get brightest 1% of pixels -> halo around hologram center
            threshold = np.percentile(image_np, 99.0)
            bright_core = image_np > threshold

            # calculate hologram center coordinates
            center_y, center_x = scipy.ndimage.center_of_mass(bright_core)
            if np.isnan(center_y) or np.isnan(center_x):
                cy, cx = image.shape[1] // 2, image.shape[2] // 2
            else:
                cy, cx = int(round(center_y)), int(round(center_x))

            _, h, w = image.shape
            shift_y = (h // 2) - cy
            shift_x = (w // 2) - cx

            image = torch.roll(image, shifts=(shift_y, shift_x), dims=(1, 2))
            mask = torch.roll(mask, shifts=(shift_y, shift_x), dims=(1, 2))

        image = TF.center_crop(image, output_size=[self.crop_size, self.crop_size])
        mask = TF.center_crop(mask, output_size=[self.crop_size, self.crop_size])

        # apply adaptive pooling
        image = F.adaptive_avg_pool2d(image, self.target_size)
        mask = F.adaptive_max_pool2d(mask, self.target_size)

        return image, mask


class HologramDataset(Dataset):
    def __init__(
        self,
        data_samples: list,
        label_map: dict,
        transform=None,
        mode: str = "raw",
        add_poisson_noise: bool = False,
    ):
        """
        :param data_samples: list of dicts [{'filepath': str, 'key': str, 'label': str}]
        :param label_map: dict mapping string labels to integers
        """
        self.data_samples = data_samples
        self.label_map = label_map
        self.transform = transform
        self.mode = mode
        self.add_poisson_noise = add_poisson_noise

    def __len__(self):
        if self.mode == "raw":
            return len(self.data_samples) * 2
        else:
            return len(self.data_samples)

    def _scale_hologram(self, holo, is_diff=False):
        if is_diff:
            holo = np.sign(holo) * np.log1p(np.abs(holo))
            max_abs = np.max(np.abs(holo))
            if max_abs > 0:
                holo = holo / max_abs
        else:
            holo = np.clip(holo, 0, None)
            holo = np.log1p(holo)
            h_min = holo.min()
            h_max = holo.max()
            if h_max > h_min:
                holo = (holo - h_min) / (h_max - h_min)
            else:
                holo = holo - h_min
        return holo

    def __getitem__(self, idx):
        run_idx = idx // 2 if self.mode == "raw" else idx
        sample_info = self.data_samples[run_idx]

        filepath = sample_info["filepath"]
        run_key = sample_info["key"]
        label_str = sample_info["label"]

        with h5py.File(filepath, "r") as h5_file:
            run_data = h5_file[run_key]

            holo_cl = np.squeeze(run_data["CL"]["detected"][:])
            holo_cr = np.squeeze(run_data["CR"]["detected"][:])
            mask_np = np.squeeze(run_data["beamstop_mask"][:])

        if self.add_poisson_noise:
            holo_cl = np.random.poisson(np.clip(holo_cl, 0, None)).astype(np.float32)
            holo_cr = np.random.poisson(np.clip(holo_cr, 0, None)).astype(np.float32)

        if self.mode == "rgb":
            holo_diff = holo_cl - holo_cr
            holo_cl = self._scale_hologram(holo_cl, is_diff=False)
            holo_cr = self._scale_hologram(holo_cr, is_diff=False)
            holo_diff = self._scale_hologram(holo_diff, is_diff=True)
            tensor = torch.from_numpy(
                np.stack([holo_cl, holo_cr, holo_diff], axis=0)
            ).float()

        elif self.mode == "diff":
            holo_diff = holo_cl - holo_cr
            holo_diff = self._scale_hologram(holo_diff, is_diff=True)
            tensor = torch.from_numpy(holo_diff).float().unsqueeze(0)

        elif self.mode == "raw":
            is_cr = idx % 2
            holo = holo_cr if is_cr else holo_cl
            holo = self._scale_hologram(holo, is_diff=False)
            tensor = torch.from_numpy(holo).float().unsqueeze(0)
        else:
            raise ValueError(f"Unknown mode specified: {self.mode}")

        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        if self.transform:
            tensor, mask_tensor = self.transform(tensor, mask_tensor)

        label_tensor = torch.tensor(self.label_map[label_str], dtype=torch.long)
        return tensor, label_tensor, mask_tensor


class HologramDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = min(
            6,
            max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) - 1),
        ),
        center_holograms: bool = True,
        mode: str = "rgb",
        add_poisson_noise: bool = False,
        limit_samples: int = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 224
        self.initial_crop_size = 960
        self.add_poisson_noise = add_poisson_noise
        self.mode = mode
        self.limit_samples = limit_samples
        self.setup_loaded = False

        self.transform = CDICropAndBinTransform(
            crop_size=self.initial_crop_size,
            target_size=self.img_size,
            center_holograms=center_holograms,
        )

    def _build_or_load_index(self):
        """Scans HDF5 files once and creates a CSV index or loads it if it already exists."""
        index_file = self.data_dir / "dataset_index.csv"

        if index_file.exists():
            logger.info(f"Loading metadata index from {index_file}")
            df = pd.read_csv(index_file, dtype={"key": str})
            return df.to_dict("records")

        logger.info(
            f"No index found. Scanning HDF5 files in {self.data_dir} (This happens only once)..."
        )
        h5_files = sorted(glob.glob(str(self.data_dir / "*.h5")))

        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {self.data_dir}")

        data_samples = []
        for file_path in tqdm(h5_files, desc="Indexing H5 files"):
            try:
                with h5py.File(file_path, "r") as f:
                    run_keys = [
                        k for k in f.keys() if k != "_pipeline_config" and k.isdigit()
                    ]
                    for key in run_keys:
                        meta = f[key]["metadata"]
                        if "magnetic_pattern" in meta.keys():
                            lbl = meta["magnetic_pattern"]["pattern_type_method"][()]
                        else:
                            lbl = meta["sample"]["magnetic_pattern"][
                                "pattern_type_method"
                            ][()]

                        if isinstance(lbl, bytes):
                            lbl = lbl.decode("utf-8")

                        data_samples.append(
                            {"filepath": file_path, "key": key, "label": lbl}
                        )
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        # save to cache for next time
        df = pd.DataFrame(data_samples)
        df.to_csv(index_file, index=False)
        logger.info(f"Saved dataset index to {index_file}")

        return data_samples

    def setup(self, stage=None):
        if self.setup_loaded:
            return

        all_samples = self._build_or_load_index()
        run_labels = [s["label"] for s in all_samples]
        unique_labels = sorted(list(set(run_labels)))
        self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}

        logger.info(f"Found {len(all_samples)} total runs. Classes: {self.label_map}")

        # generate stratified splits
        if self.limit_samples is not None and self.limit_samples < len(all_samples):
            logger.info(f"Reducing dataset to {self.limit_samples} instances.")
            all_samples, _, run_labels, _ = train_test_split(
                all_samples,
                run_labels,
                train_size=self.limit_samples,
                stratify=run_labels,
                random_state=42,
            )

        train_samples, temp_samples, train_labels, temp_labels = train_test_split(
            all_samples, run_labels, test_size=0.2, stratify=run_labels, random_state=42
        )
        val_samples, test_samples, _, _ = train_test_split(
            temp_samples,
            temp_labels,
            test_size=0.5,
            stratify=temp_labels,
            random_state=42,
        )

        self.train_dataset = HologramDataset(
            train_samples,
            self.label_map,
            transform=self.transform,
            mode=self.mode,
            add_poisson_noise=self.add_poisson_noise,
        )
        self.val_dataset = HologramDataset(
            val_samples,
            self.label_map,
            transform=self.transform,
            mode=self.mode,
            add_poisson_noise=self.add_poisson_noise,
        )
        self.test_dataset = HologramDataset(
            test_samples,
            self.label_map,
            transform=self.transform,
            mode=self.mode,
            add_poisson_noise=self.add_poisson_noise,
        )

        self.setup_loaded = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    data_path = "C:/Users/kelle/Documents/storage/xray/Raw_holo_sim"
    current_mode = "rgb"  # options: rgb, raw, diff
    current_noise = True

    data_module = HologramDataModule(
        data_dir=data_path,
        batch_size=4,
        num_workers=0,
        mode=current_mode,
        center_holograms=True,
        add_poisson_noise=current_noise,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    inv_label_map = {v: k for k, v in data_module.label_map.items()}

    class ViewerState:
        def __init__(self, dataloader, inv_map, mode):
            self.dataloader = dataloader
            self.data_iter = iter(self.dataloader)
            self.inv_map = inv_map
            self.mode = mode
            self.current_batch = next(self.data_iter)
            self.batch_idx = 0
            self.batch_size = self.current_batch[0].shape[0]

        def next_image(self, event):
            self.batch_idx += 1
            if self.batch_idx >= self.batch_size:
                try:
                    self.current_batch = next(self.data_iter)
                    self.batch_idx = 0
                    self.batch_size = self.current_batch[0].shape[0]
                except StopIteration:
                    self.data_iter = iter(self.dataloader)
                    self.current_batch = next(self.data_iter)
                    self.batch_idx = 0

            self.update_plot()

        def update_plot(self):
            holo, label, mask = self.current_batch
            idx = self.batch_idx
            m = mask[idx].squeeze(0).numpy()

            class_idx = label[idx].item()
            class_name = self.inv_map.get(class_idx, "Unknown")

            if self.mode == "rgb":
                h_cl = holo[idx][0].squeeze().numpy()
                h_cr = holo[idx][1].squeeze().numpy()
                h_diff = holo[idx][2].squeeze().numpy()

                img_cl.set_data(h_cl)
                img_cl.set_clim(vmin=h_cl.min(), vmax=h_cl.max())

                img_cr.set_data(h_cr)
                img_cr.set_clim(vmin=h_cr.min(), vmax=h_cr.max())

                img_diff.set_data(h_diff)
                max_val = max(abs(h_diff.min()), abs(h_diff.max()))
                img_diff.set_clim(vmin=-max_val, vmax=max_val)
            else:
                h_single = holo[idx][0].squeeze().numpy()
                img_single.set_data(h_single)

                if self.mode == "diff":
                    max_val = max(abs(h_single.min()), abs(h_single.max()))
                    img_single.set_clim(vmin=-max_val, vmax=max_val)
                else:
                    img_single.set_clim(vmin=h_single.min(), vmax=h_single.max())

            img_mask.set_data(m)
            fig.suptitle(
                f"Hologram Overview | Class: {class_name}",
                fontsize=14,
                fontweight="bold",
            )
            fig.canvas.draw_idle()

    viewer = ViewerState(train_loader, inv_label_map, data_module.mode)

    holo_init, label_init, mask_init_batch = viewer.current_batch
    mask_init = mask_init_batch[0].squeeze(0).numpy()
    init_class = inv_label_map.get(label_init[0].item(), "Unknown")

    if data_module.mode == "rgb":
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 6))
        plt.subplots_adjust(bottom=0.25)

        init_cl = holo_init[0][0].squeeze().numpy()
        init_cr = holo_init[0][1].squeeze().numpy()
        init_diff = holo_init[0][2].squeeze().numpy()

        img_cl = ax1.imshow(init_cl, cmap="viridis")
        ax1.set_title("Kanal 0: CL")
        fig.colorbar(img_cl, ax=ax1, fraction=0.046, pad=0.04)

        img_cr = ax2.imshow(init_cr, cmap="viridis")
        ax2.set_title("Kanal 1: CR")
        fig.colorbar(img_cr, ax=ax2, fraction=0.046, pad=0.04)

        img_diff = ax3.imshow(init_diff, cmap="coolwarm")
        ax3.set_title("Kanal 2: Diff (CL - CR)")
        max_init_val = max(abs(init_diff.min()), abs(init_diff.max()))
        img_diff.set_clim(vmin=-max_init_val, vmax=max_init_val)
        fig.colorbar(img_diff, ax=ax3, fraction=0.046, pad=0.04)

        img_mask = ax4.imshow(mask_init, cmap="gray")
        ax4.set_title("Beamstop Maske")
        fig.colorbar(img_mask, ax=ax4, fraction=0.046, pad=0.04)

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)

        init_single = holo_init[0][0].squeeze().numpy()
        cmap = "coolwarm" if data_module.mode == "diff" else "viridis"
        img_single = ax1.imshow(init_single, cmap=cmap)
        ax1.set_title("Hologram (RAW)" if data_module.mode == "raw" else "Diff-Holo")

        if data_module.mode == "diff":
            max_init_val = max(abs(init_single.min()), abs(init_single.max()))
            img_single.set_clim(vmin=-max_init_val, vmax=max_init_val)
        else:
            img_single.set_clim(vmin=init_single.min(), vmax=init_single.max())

        fig.colorbar(img_single, ax=ax1, fraction=0.046, pad=0.04)

        img_mask = ax2.imshow(mask_init, cmap="gray")
        ax2.set_title("Beamstop Maske")
        fig.colorbar(img_mask, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Hologram Overview - Class: {init_class}", fontsize=14, fontweight="bold"
    )
    ax_button = plt.axes([0.45, 0.05, 0.1, 0.06])
    btn_next = Button(ax_button, "Next")
    btn_next.on_clicked(viewer.next_image)

    plt.show()
