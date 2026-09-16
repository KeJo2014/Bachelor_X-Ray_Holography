import os
import io
import json
import glob
import torch
import random
import logging
import tifffile
import numpy as np
import webdataset as wds
import random
from pathlib import Path
from pytorch_lightning import LightningDataModule

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "binary_labyrinth_pattern": 0,
    "saturated_pattern": 1,
    "disordered_skyrmion_lattice_pattern": 2,
}


def decode_stream(data_iterator, mode="rgb", label_map=None):
    """
    Decodes a webdataset stream and yields tensors, labels, and masks.
    """

    for sample in data_iterator:
        meta = json.loads(sample["json"])

        lbl_str = meta["metadata"]["sample"]["magnetic_pattern"]["pattern_type_method"]
        label_idx = label_map.get(lbl_str, 0)
        label = torch.tensor(label_idx, dtype=torch.long)

        cl = tifffile.imread(io.BytesIO(sample["cl.tiff"])).astype(np.float32)
        cr = tifffile.imread(io.BytesIO(sample["cr.tiff"])).astype(np.float32)
        mask = tifffile.imread(io.BytesIO(sample["beamstop_mask.tiff"])).astype(
            np.float32
        )

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if mode in ["rgb", "diff"]:
            tensor = torch.from_numpy(np.stack([cl, cr], axis=0))
            yield tensor, label, mask_tensor

        elif mode == "raw":
            tensor_cl = torch.from_numpy(cl).unsqueeze(0)
            yield tensor_cl, label, mask_tensor

            tensor_cr = torch.from_numpy(cr).unsqueeze(0)
            yield tensor_cr, label, mask_tensor

        else:
            raise ValueError(f"Unknown mode specified: {mode}")


class HologramDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        total_samples: int,
        batch_size: int = 32,
        num_workers: int = min(
            12,
            max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) - 1),
        ),
        mode: str = "rgb",
        add_poisson_noise: bool = False,
        prefetch_factor: int = 6,
        train_fraction: float = 1.0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_poisson_noise = add_poisson_noise
        self.mode = mode
        self.prefetch_factor = prefetch_factor
        self.label_map = LABEL_MAP
        self.setup_loaded = False
        self.total_samples = total_samples
        self.train_fraction = train_fraction

        if mode == "raw":
            self.total_samples *= 2

    def setup(self, stage=None):
        """
        Handles the setup of the dataset, including splitting into train, validation, and test sets.
        """

        if self.setup_loaded:
            return

        raw_shards = sorted(glob.glob(str(self.data_dir / "*.tar")))
        random.Random(42).shuffle(raw_shards)
        if not raw_shards:
            raise FileNotFoundError(f"No tar files in {self.data_dir} found.")
        shards = [f"file:{Path(p).as_posix()}" for p in raw_shards]

        num_shards = len(shards)
        # generate split on archive level
        train_end = int(0.8 * num_shards)
        val_end = int(0.9 * num_shards)
        full_train_urls = shards[:train_end]

        # option to reduce labeled train size
        num_train_shards = (
            max(1, int(len(full_train_urls) * self.train_fraction))
            if self.train_fraction > 0
            else 0
        )
        self.train_urls = full_train_urls[:num_train_shards]
        self.val_urls = shards[train_end:val_end]
        self.test_urls = shards[val_end:]

        if num_shards > 0:
            self.train_samples = int(
                self.total_samples * (len(self.train_urls) / num_shards)
            )
        else:
            self.train_samples = 0

        logger.info(
            f"Shards distributed: Train={len(self.train_urls)} (out of originally {len(full_train_urls)}), "
            f"Val={len(self.val_urls)}, Test={len(self.test_urls)}"
        )
        self.setup_loaded = True

    def _create_dataset(self, urls, is_train=False):
        """
        Creates a WebDataset dataset from the provided URLs.
        """

        if not urls:
            raise ValueError(f"No urls received!")

        dataset = wds.WebDataset(
            urls,
            resampled=is_train,
            nodesplitter=wds.split_by_node,
            shardshuffle=False,
            empty_check=False,
        )

        if is_train:
            dataset = dataset.shuffle(1000)
        dataset = dataset.compose(
            lambda it: decode_stream(it, self.mode, self.label_map)
        )
        if is_train and self.mode == "raw":
            dataset = dataset.shuffle(100)

        dataset = dataset.batched(self.batch_size, partial=not is_train)

        if is_train and self.train_samples > 0:
            batches_per_epoch = self.train_samples // self.batch_size
            dataset = dataset.with_epoch(batches_per_epoch).with_length(
                batches_per_epoch
            )
        return dataset

    def train_dataloader(self):
        if not self.train_urls:
            return None
        dataset = self._create_dataset(self.train_urls, is_train=True)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        dataset = self._create_dataset(self.val_urls, is_train=False)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        dataset = self._create_dataset(self.test_urls, is_train=False)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Post-processing after batch transfer to device.
        """

        raw_tensor, labels, masks = batch

        def scale_gpu(holo, is_diff=False):
            if is_diff:
                signs = torch.sign(holo)
                holo = holo.abs_().log1p_()
                holo = signs * holo
                max_abs = torch.amax(holo.abs(), dim=(-2, -1), keepdim=True)
                holo = torch.where(max_abs > 0, holo / max_abs, holo)
            else:
                holo = holo.clamp_(min=0).log1p_()
                h_min = torch.amin(holo, dim=(-2, -1), keepdim=True)
                h_max = torch.amax(holo, dim=(-2, -1), keepdim=True)
                denominator = torch.clamp(h_max - h_min, min=1e-8)
                holo = (holo - h_min) / denominator
            return holo

        if self.add_poisson_noise:
            raw_tensor = torch.poisson(raw_tensor.clamp_(min=0))

        if self.mode == "rgb":
            t_cl = scale_gpu(raw_tensor[:, 0:1], is_diff=False)
            t_cr = scale_gpu(raw_tensor[:, 1:2], is_diff=False)
            t_diff = scale_gpu(raw_tensor[:, 0:1] - raw_tensor[:, 1:2], is_diff=True)
            tensor = torch.cat([t_cl, t_cr, t_diff], dim=1)
        elif self.mode == "diff":
            tensor = scale_gpu(raw_tensor[:, 0:1] - raw_tensor[:, 1:2], is_diff=True)
        elif self.mode == "raw":
            tensor = scale_gpu(raw_tensor, is_diff=False)

        return tensor, labels, masks


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np

    data_path = "C:/Users/kelle/Documents/storage/xray/Raw_holo_sim/reduced2"
    current_mode = "rgb"
    current_noise = False

    data_module = HologramDataModule(
        data_dir=data_path,
        total_samples=5000,
        batch_size=4,
        num_workers=0,
        mode=current_mode,
        add_poisson_noise=current_noise,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    inv_label_map = {v: k for k, v in data_module.label_map.items()}

    def cpu_scale_for_plot(holo, is_diff=False):
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

    class ViewerState:
        def __init__(self, dataloader, inv_map, mode):
            self.dataloader = dataloader
            self.data_iter = iter(self.dataloader)
            self.inv_map = inv_map
            self.mode = mode

            raw_batch = next(self.data_iter)
            self.current_batch = data_module.on_after_batch_transfer(raw_batch, 0)

            self.batch_idx = 0
            self.batch_size = self.current_batch[0].shape[0]

        def next_image(self, event):
            self.batch_idx += 1
            if self.batch_idx >= self.batch_size:
                try:
                    raw_batch = next(self.data_iter)
                    self.current_batch = data_module.on_after_batch_transfer(
                        raw_batch, 0
                    )
                    self.batch_idx = 0
                    self.batch_size = self.current_batch[0].shape[0]
                except StopIteration:
                    self.data_iter = iter(self.dataloader)
                    raw_batch = next(self.data_iter)
                    self.current_batch = data_module.on_after_batch_transfer(
                        raw_batch, 0
                    )
                    self.batch_idx = 0
            self.update_plot()

        def update_plot(self):
            holo, label, _ = self.current_batch
            idx = self.batch_idx
            class_idx = label[idx].item()
            class_name = self.inv_map.get(class_idx, "Unknown")
            channels = holo.shape[1]

            for c in range(channels):
                is_diff_channel = (self.mode == "rgb" and c == 2) or (
                    self.mode == "diff"
                )

                raw_data = holo[idx][c].cpu().numpy()

                h_single = cpu_scale_for_plot(raw_data, is_diff=is_diff_channel)
                ifft_data = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(raw_data)))
                ifft_mag_scaled = cpu_scale_for_plot(np.abs(ifft_data), is_diff=False)

                img_raws[c].set_data(h_single)
                img_iffts[c].set_data(ifft_mag_scaled)

                vmin = -1 if is_diff_channel else 0
                img_raws[c].set_clim(vmin=vmin, vmax=1)
                img_iffts[c].set_clim(vmin=0, vmax=1)

            fig.suptitle(
                f"CL Raw Hologram & IFFT | Class: {class_name}",
                fontsize=14,
                fontweight="bold",
            )
            fig.canvas.draw_idle()

    viewer = ViewerState(train_loader, inv_label_map, data_module.mode)
    holo_init, label_init, _ = viewer.current_batch
    init_class = inv_label_map.get(label_init[0].item(), "Unknown")

    channels = holo_init.shape[1]

    fig, axes = plt.subplots(channels, 2, figsize=(12, 4 * channels))
    plt.subplots_adjust(bottom=0.15 / channels + 0.05, hspace=0.3)

    if channels == 1:
        axes = np.expand_dims(axes, axis=0)

    img_raws = []
    img_iffts = []

    if current_mode == "rgb" and channels == 3:
        channel_names = ["CL", "CR", "Diff (Magnetic)"]
    elif current_mode == "diff":
        channel_names = ["Diff (Magnetic)"]
    else:
        channel_names = [f"Channel {c}" for c in range(channels)]

    for c in range(channels):
        is_diff_channel = (current_mode == "rgb" and c == 2) or (current_mode == "diff")

        init_raw = holo_init[0][c].cpu().numpy()
        init_single = cpu_scale_for_plot(init_raw, is_diff=is_diff_channel)

        init_ifft = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(init_raw)))
        init_ifft_mag_scaled = cpu_scale_for_plot(np.abs(init_ifft), is_diff=False)

        cmap_raw = "bwr" if is_diff_channel else "gray"
        vmin_raw = -1 if is_diff_channel else 0

        # plot CL/CR/Diff raw hologram
        ax_raw = axes[c, 0]
        img_single = ax_raw.imshow(init_single, cmap=cmap_raw, vmin=vmin_raw, vmax=1)
        ax_raw.set_title(f"Raw Hologram ({channel_names[c]})")
        fig.colorbar(img_single, ax=ax_raw, fraction=0.046, pad=0.04)
        img_raws.append(img_single)

        # plot IFFT magnitude
        ax_ifft = axes[c, 1]
        img_ifft = ax_ifft.imshow(init_ifft_mag_scaled, cmap="gray", vmin=0, vmax=1)
        ax_ifft.set_title(f"IFFT Magnitude ({channel_names[c]})")
        fig.colorbar(img_ifft, ax=ax_ifft, fraction=0.046, pad=0.04)
        img_iffts.append(img_ifft)

    fig.suptitle(
        f"Hologram & IFFT | Class: {init_class}",
        fontsize=14,
        fontweight="bold",
    )

    ax_button = plt.axes([0.45, 0.02, 0.1, 0.05])
    btn_next = Button(ax_button, "Next")
    btn_next.on_clicked(viewer.next_image)

    plt.show()
