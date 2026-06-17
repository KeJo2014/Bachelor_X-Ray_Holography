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
            image_np = image.squeeze(0).numpy()
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
        use_difference_holograms: bool = False,
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
        self.use_difference_holograms = use_difference_holograms
        self.h5_file = None

    def open_hdf5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_filepath, "r")

    def __len__(self):
        if not self.use_difference_holograms:
            return len(self.run_keys) * 2  # each run yields two holograms (CL and CR)
        else:
            return len(self.run_keys)  # CL and CR are used to calculate the diff holo

    def __getitem__(self, idx):
        self.open_hdf5()

        run_idx = idx // 2
        run_key = self.run_keys[run_idx]
        run_data = self.h5_file[run_key]

        label_val = run_data["metadata"]["sample"]["magnetic_pattern"][
            "pattern_type_method"
        ][()].decode("utf-8")

        # Load hologram
        holo = None
        if self.use_difference_holograms:
            holo = (run_data["CL"]["detected"][:]) - (run_data["CR"]["detected"][:])
        else:
            is_cr = idx % 2
            holo = (
                run_data["CR"]["detected"][:]
                if is_cr
                else run_data["CL"]["detected"][:]
            )

        mask_np = run_data["beamstop_mask"][:]
        holo = np.squeeze(holo).astype(np.float32)
        mask_np = np.squeeze(mask_np)

        if self.use_difference_holograms:
            # use symmetric logarithm for difference holograms to maintain the sign and compress dynamic
            holo = np.sign(holo) * np.log1p(np.abs(holo))

            # zero preserving normalization (range [-1,1])
            max_abs = np.max(np.abs(holo))
            if max_abs > 0:
                holo = holo / max_abs
        else:
            holo = np.clip(holo, 0, None)
            holo = np.log1p(holo)
            h_max = holo.max()
            if h_max > 0:
                holo = holo / h_max

        # convert to pytorch tensor with dim [1, H, W]
        tensor = torch.from_numpy(holo).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        # if applicable apply transformation
        if self.transform:
            tensor, mask_tensor = self.transform(tensor, mask_tensor)

        # generate one-hot-tensor map
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
        use_difference_holograms: bool = False,
        center_holograms: bool = True,
    ):
        super().__init__(
            data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
        )
        self.img_size = 960
        self.setup_loaded = False
        self.use_difference_holograms = use_difference_holograms
        self.initial_crop_size = 1100  # TODO: renove if ot necessary anymore

        self.transform = CDICropAndBinTransform(
            crop_size=self.initial_crop_size,
            target_size=self.img_size,
            center_holograms=center_holograms,
        )

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
                lbl = data[key]["metadata"]["sample"]["magnetic_pattern"][
                    "pattern_type_method"
                ][()]
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

            if self.use_difference_holograms:
                logger.info(
                    f"Found {len(run_keys)} total samples from {len(run_keys)} runs. Calculating Difference Holograms"
                )
            else:
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
            use_difference_holograms=self.use_difference_holograms,
        )
        self.val_dataset = HologramDataset(
            self.data_dir,
            val_keys,
            label_map,
            transform=self.transform,
            use_difference_holograms=self.use_difference_holograms,
        )
        self.test_dataset = HologramDataset(
            self.data_dir,
            test_keys,
            label_map,
            transform=self.transform,
            use_difference_holograms=self.use_difference_holograms,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    # Initialize data module
    data_path = (
        "C:\\Users\\kelle\\Documents\\storage\\xray\\Raw_holo_sim\\master_dataset.h5"
    )
    data_module = HologramDataModule(
        data_path, use_difference_holograms=False, center_holograms=True
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    inv_label_map = {v: k for k, v in data_module.train_dataset.label_map.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    class ViewerState:
        def __init__(self, dataloader, inv_map):
            self.dataloader = dataloader
            self.data_iter = iter(self.dataloader)
            self.inv_map = inv_map
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
            h = holo[idx].squeeze(0).numpy()
            m = mask[idx].squeeze(0).numpy()

            class_idx = torch.argmax(label[idx]).item()
            class_name = self.inv_map[class_idx]

            img1.set_data(h)
            img1.set_clim(vmin=h.min(), vmax=h.max())
            ax1.set_title(f"Hologram | Label: {class_name}")

            img2.set_data(m)
            ax2.set_title("Beamstop Mask")

            fig.canvas.draw_idle()

    viewer = ViewerState(train_loader, inv_label_map)
    holo_init = viewer.current_batch[0][0].squeeze(0).numpy()
    mask_init = viewer.current_batch[2][0].squeeze(0).numpy()
    init_class = inv_label_map[torch.argmax(viewer.current_batch[1][0]).item()]

    # initial images
    img1 = ax1.imshow(holo_init, cmap="viridis")
    ax1.set_title(f"Hologram | Label: {init_class}")
    fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)

    img2 = ax2.imshow(mask_init, cmap="gray")
    ax2.set_title("Beamstop Mask")
    fig.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)

    ax_button = plt.axes([0.45, 0.05, 0.1, 0.075])
    btn_next = Button(ax_button, "Next")
    btn_next.on_clicked(viewer.next_image)

    plt.show()
