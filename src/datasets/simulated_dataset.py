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
        num_train_shards = max(1, int(len(full_train_urls) * self.train_fraction)) if self.train_fraction > 0 else 0
        self.train_urls = full_train_urls[:num_train_shards]
        self.val_urls = shards[train_end:val_end]
        self.test_urls = shards[val_end:]

        if num_shards > 0:
            self.train_samples = int(self.total_samples * (len(self.train_urls) / num_shards))
        else:
            self.train_samples = 0

        logger.info(
            f"Shards distributed: Train={len(self.train_urls)} (out of originally {len(full_train_urls)}), "
            f"Val={len(self.val_urls)}, Test={len(self.test_urls)}"
        )
        self.setup_loaded = True

    def _create_dataset(self, urls, is_train=False):
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

    data_path = "C:/Users/kelle/Documents/storage/xray/Raw_holo_sim/reduced2"
    current_mode = "raw"
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
            holo, label, mask = self.current_batch
            idx = self.batch_idx
            m = mask[idx].squeeze(0).cpu().numpy()
            class_idx = label[idx].item()
            class_name = self.inv_map.get(class_idx, "Unknown")

            if self.mode == "rgb":
                cl_raw = holo[idx][0].cpu().numpy()
                cr_raw = holo[idx][1].cpu().numpy()
                h_cl = cpu_scale_for_plot(cl_raw, False)
                h_cr = cpu_scale_for_plot(cr_raw, False)
                h_diff = cpu_scale_for_plot(cl_raw - cr_raw, True)

                img_cl.set_data(h_cl)
                img_cl.set_clim(vmin=h_cl.min(), vmax=h_cl.max())
                img_cr.set_data(h_cr)
                img_cr.set_clim(vmin=h_cr.min(), vmax=h_cr.max())
                img_diff.set_data(h_diff)
                max_val = max(abs(h_diff.min()), abs(h_diff.max()))
                img_diff.set_clim(vmin=-max_val, vmax=max_val)
            else:
                h_single = cpu_scale_for_plot(
                    holo[idx][0].cpu().numpy(), is_diff=(self.mode == "diff")
                )
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
    mask_init = mask_init_batch[0].squeeze(0).cpu().numpy()
    init_class = inv_label_map.get(label_init[0].item(), "Unknown")

    if data_module.mode == "rgb":
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 6))
        plt.subplots_adjust(bottom=0.25)

        init_cl_raw = holo_init[0][0].cpu().numpy()
        init_cr_raw = holo_init[0][1].cpu().numpy()
        init_cl = cpu_scale_for_plot(init_cl_raw, False)
        init_cr = cpu_scale_for_plot(init_cr_raw, False)
        init_diff = cpu_scale_for_plot(init_cl_raw - init_cr_raw, True)

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

        init_single = cpu_scale_for_plot(
            holo_init[0][0].cpu().numpy(), is_diff=(data_module.mode == "diff")
        )
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
