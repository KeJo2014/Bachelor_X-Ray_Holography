import os
import torch
import numpy as np
import logging
import h5py
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import scipy.ndimage
from torch.utils.data import Dataset, DataLoader
from datasets.abstract_dataset import AbstractDataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CDICropAndBinTransform:
    """
    Computes central crop with average pixel binning for CDI data
    Use average for diffration image and max pooling for the beamstop mask
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

            # calculate hologram center coordinates and calc crop coordinates
            center_y, center_x = scipy.ndimage.center_of_mass(bright_core)
            cy = int(round(center_y))
            cx = int(round(center_x))
            crop_h, crop_w = self.crop_size, self.crop_size
            top = cy - (crop_h // 2)
            left = cx - (crop_w // 2)

            image = TF.crop(image, top, left, crop_h, crop_w)
            mask = TF.crop(mask, top, left, crop_h, crop_w)
        else:
            image = TF.center_crop(image, output_size=[self.crop_size, self.crop_size])
            mask = TF.center_crop(mask, output_size=[self.crop_size, self.crop_size])

        # apply adaptiv pooling
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
        mode: str = "raw",
        add_poisson_noise: bool = False,
    ):
        """
        :param h5_filepath: path to the h5 data file
        :param run_keys: list of specific h5 run keys assigned to this split
        :param label_map: dict mapping labels to class integers
        :param transform: pytorch transformation
        :param mode: "raw" (1-chan CL/CR), "diff" (1-chan Diff), or "rgb" (3-chan CL, CR, Diff)
        :param add_poisson_noise: If true, applies shot noise to raw photon counts
        """
        self.h5_filepath = h5_filepath
        self.run_keys = run_keys
        self.label_map = label_map
        self.transform = transform
        self.mode = mode
        self.add_poisson_noise = add_poisson_noise
        self.h5_file = None

    def open_hdf5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_filepath, "r")

    def __len__(self):
        if self.mode == "raw":
            return len(self.run_keys) * 2  # CL and CR treated as individual samples
        else:
            return len(self.run_keys)  # Diff or RGB combine CL and CR into one sample

    def _scale_hologram(self, holo, is_diff=False):
        if is_diff:
            # symmetric logarithm for difference holograms
            holo = np.sign(holo) * np.log1p(np.abs(holo))
            max_abs = np.max(np.abs(holo))
            if max_abs > 0:
                holo = holo / max_abs
        else:
            # min-max scaling for raw holograms
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
        self.open_hdf5()

        run_idx = idx // 2 if self.mode == "raw" else idx
        run_key = self.run_keys[run_idx]
        run_data = self.h5_file[run_key]

        if "magnetic_pattern" in run_data["metadata"].keys():
            label_val = run_data["metadata"]["magnetic_pattern"]["pattern_type_method"][
                ()
            ].decode("utf-8")
        else:
            label_val = run_data["metadata"]["sample"]["magnetic_pattern"][
                "pattern_type_method"
            ][()].decode("utf-8")

        holo_cl = np.squeeze(run_data["CL"]["detected"][:])
        holo_cr = np.squeeze(run_data["CR"]["detected"][:])

        # apply poisson shot noise
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

        mask_np = np.squeeze(run_data["beamstop_mask"][:])
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        if self.transform:
            tensor, mask_tensor = self.transform(tensor, mask_tensor)

        num_classes = len(self.label_map)
        label_tensor = torch.zeros(num_classes, dtype=torch.float32)
        label_tensor[self.label_map[label_val]] = 1.0

        return tensor, label_tensor, mask_tensor


class HologramDataModule(AbstractDataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = min(
            13,
            max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) - 1),
        ),
        center_holograms: bool = True,
        mode: str = None,
        add_poisson_noise: bool = False,
        limit_samples: int = None,
    ):
        super().__init__(
            data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
        )
        self.img_size = 224
        self.setup_loaded = False
        self.initial_crop_size = 500
        self.add_poisson_noise = add_poisson_noise
        self.mode = mode
        self.limit_samples = limit_samples

        self.transform = CDICropAndBinTransform(
            crop_size=self.initial_crop_size,
            target_size=self.img_size,
            center_holograms=center_holograms,
        )

    def setup(self, stage=None):
        if self.setup_loaded:
            return

        with h5py.File(self.data_dir, "r") as f:
            data = f
            run_keys = [k for k in f.keys() if k != "_pipeline_config"]

            # find all unique classes
            all_labels = set()
            run_labels = []
            for key in run_keys:
                if "magnetic_pattern" in data[key]["metadata"].keys():
                    lbl = data[key]["metadata"]["magnetic_pattern"][
                        "pattern_type_method"
                    ][()]
                else:
                    lbl = data[key]["metadata"]["sample"]["magnetic_pattern"][
                        "pattern_type_method"
                    ][()]
                if isinstance(lbl, bytes):
                    lbl = lbl.decode("utf-8")
                all_labels.add(lbl)
                run_labels.append(lbl)

                assert "CL" in data[key].keys()
                assert "CR" in data[key].keys()

            # create map of all unique labels
            unique_labels = sorted(list(all_labels))
            label_map = {lbl: i for i, lbl in enumerate(unique_labels)}

            if self.mode != "raw":
                logger.info(
                    f"Found {len(run_keys)} total samples from {len(run_keys)} runs. Mode: {self.mode.upper()} | Poisson Noise: {self.add_poisson_noise}"
                )
            else:
                logger.info(
                    f"Found {len(run_keys)*2} total samples from {len(run_keys)} runs. Mode: RAW | Poisson Noise: {self.add_poisson_noise}"
                )
            logger.info(f"Detected classes: {label_map}")

            # if specified select stratified datasubset before splitting data
            if self.limit_samples != None and self.limit_samples < len(run_keys):
                logger.info(
                    f"Stratified dataset reduction of {len(run_keys)} to {self.limit_samples} instances."
                )
                run_keys, _, run_labels, _ = train_test_split(
                    run_keys,
                    run_labels,
                    train_size=self.limit_samples,
                    stratify=run_labels,
                    random_state=42,
                )

            # stratified group shuffle split
            train_keys, temp_keys, train_labels, temp_labels = train_test_split(
                run_keys,
                run_labels,
                test_size=0.2,
                stratify=run_labels,
                random_state=42,
            )
            val_keys, test_keys, _, _ = train_test_split(
                temp_keys,
                temp_labels,
                test_size=0.5,
                stratify=temp_labels,
                random_state=42,
            )

        # instantiate datasets
        self.train_dataset = HologramDataset(
            self.data_dir,
            train_keys,
            label_map,
            transform=self.transform,
            mode=self.mode,
            add_poisson_noise=self.add_poisson_noise,
        )
        self.val_dataset = HologramDataset(
            self.data_dir,
            val_keys,
            label_map,
            transform=self.transform,
            mode=self.mode,
            add_poisson_noise=self.add_poisson_noise,
        )
        self.test_dataset = HologramDataset(
            self.data_dir,
            test_keys,
            label_map,
            transform=self.transform,
            mode=self.mode,
            add_poisson_noise=self.add_poisson_noise,
        )

        self.setup_loaded = True

    def train_dataloader(self):  # TODO: überführe diese Klassen in abstract class
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    # Initialize data module
    data_path = "C:\\Users\\kelle\\Documents\\storage\\xray\\Raw_holo_sim\\simulation_sweep_1.h5"
    current_mode = "rgb"  # options: rgb, raw, diff
    current_noise = True

    data_module = HologramDataModule(
        data_path,
        mode=current_mode,
        center_holograms=True,
        add_poisson_noise=current_noise,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    inv_label_map = {v: k for k, v in data_module.train_dataset.label_map.items()}

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

            class_idx = torch.argmax(label[idx]).item()
            class_name = self.inv_map[class_idx]

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
    init_class = inv_label_map[torch.argmax(label_init[0]).item()]

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
