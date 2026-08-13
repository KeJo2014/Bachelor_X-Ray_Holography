"""
This file handles the offline preprocessing step of reading HDF5 files, apply preprocessing steps and save them in webdataset shards.
"""

import argparse
import glob
import io
import json
import tarfile
import time
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import h5py
import tifffile
from pathlib import Path
from tqdm import tqdm


class ShardWriter:
    """Handler for writing result to webdataset shards"""

    def __init__(self, pattern, maxcount=100000, maxsize=3e9):
        self.pattern = pattern
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.shard = -1
        self.tarstream = None
        self.count = 0
        self.size = 0
        self.fname = None
        self._next_stream()

    def _next_stream(self):
        if self.tarstream is not None:
            self.tarstream.close()
        self.shard += 1
        self.fname = self.pattern % self.shard
        Path(self.fname).parent.mkdir(parents=True, exist_ok=True)
        self.tarstream = tarfile.open(self.fname, "w")
        self.count = 0
        self.size = 0

    def write(self, sample):
        if self.count >= self.maxcount or self.size >= self.maxsize:
            self._next_stream()

        key = sample["__key__"]
        for name, value in sample.items():
            if name == "__key__":
                continue
            if isinstance(value, str):
                value = value.encode("utf-8")
            info = tarfile.TarInfo(name=f"{key}.{name}")
            info.size = len(value)
            info.mtime = time.time()
            self.tarstream.addfile(info, io.BytesIO(value))
            self.size += len(value)

        self.count += 1

    def close(self):
        if self.tarstream is not None:
            self.tarstream.close()
            self.tarstream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def calculate_shift(image_np):
    """Calculate hologram center shift."""
    threshold = np.percentile(image_np, 99.0)
    bright_core = image_np > threshold
    center_y, center_x = scipy.ndimage.center_of_mass(bright_core)

    if np.isnan(center_y) or np.isnan(center_x):
        cy, cx = image_np.shape[0] // 2, image_np.shape[1] // 2
    else:
        cy, cx = int(round(center_y)), int(round(center_x))

    h, w = image_np.shape
    return (h // 2) - cy, (w // 2) - cx


def transform_array(
    np_array, shift_y, shift_x, crop_size=960, target_size=224, is_mask=False
):
    """Apply roll, crop, adaptive pooling."""
    t = torch.from_numpy(np_array).float().unsqueeze(0)
    t = torch.roll(t, shifts=(shift_y, shift_x), dims=(1, 2))
    t = TF.center_crop(t, output_size=[crop_size, crop_size])

    if is_mask:
        t = F.adaptive_max_pool2d(t, (target_size, target_size))
    else:
        t = F.adaptive_avg_pool2d(t, (target_size, target_size))

    return t.squeeze(0).numpy().astype(np.float32)


def _convert_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return _convert_value(value.item())

    if isinstance(value, np.ndarray):
        return [_convert_value(v) for v in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [_convert_value(v) for v in value]

    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}

    return value


def _attr_to_native(val):
    return _convert_value(val)


def h5_to_native(obj):
    if isinstance(obj, h5py.Dataset):
        return _convert_value(obj[()])

    if isinstance(obj, h5py.Group):
        result = {}
        for key, val in obj.items():
            result[key] = h5_to_native(val)
        if obj.attrs:
            result["_attrs"] = {k: _attr_to_native(v) for k, v in obj.attrs.items()}
        return result

    return _convert_value(obj)


def array_to_tiff_bytes(array):
    buf = io.BytesIO()
    tifffile.imwrite(buf, array)
    return buf.getvalue()


def process_dataset(
    data_dir: str,
    worker_id: int,
    num_workers: int,
    maxcount: int = 1000,
    maxsize: float = 3e9,
):
    data_dir = Path(data_dir)
    reduced_dir = data_dir / "reduced"
    reduced_dir.mkdir(exist_ok=True)
    shard_pattern = str(reduced_dir / f"shard-w{worker_id:04d}-%06d.tar")
    all_h5_files = sorted(glob.glob(str(data_dir / "*.h5")))

    if not all_h5_files:
        print(f"Worker {worker_id}: No .h5 files in {data_dir} found.")
        return

    my_files = all_h5_files[worker_id::num_workers]

    print(f"Worker {worker_id}/{num_workers - 1} started.")
    print(f"Processing {len(my_files)} of {len(all_h5_files)} files.")

    if not my_files:
        print(f"Worker {worker_id}: No files to work with. Stopping.")
        return

    datasets_to_process = {
        "cl": ("CL/detected", False),
        "cr": ("CR/detected", False),
        "cl_no_beamstop": ("CL/detected_no_beamstop", False),
        "cr_no_beamstop": ("CR/detected_no_beamstop", False),
        "beamstop_mask": ("beamstop_mask", True),
    }

    with ShardWriter(shard_pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for file_path in my_files:
            filename = Path(file_path).name
            stem = Path(file_path).stem

            print(f"[Worker {worker_id}] Processing: {filename}")

            with h5py.File(file_path, "r") as f_in:
                run_keys = [
                    k for k in f_in.keys() if k != "_pipeline_config" and k.isdigit()
                ]

                pipeline_config = None
                if "_pipeline_config" in f_in.keys():
                    pipeline_config = h5_to_native(f_in["_pipeline_config"])
                desc = f"W{worker_id}: {filename}"
                for key in tqdm(run_keys, desc=desc, leave=False):
                    run_data_in = f_in[key]

                    metadata = {"source_file": filename, "run_key": key}
                    if "metadata" in run_data_in:
                        metadata["metadata"] = h5_to_native(run_data_in["metadata"])
                    if "sample" in run_data_in:
                        metadata["sample"] = h5_to_native(run_data_in["sample"])
                    if pipeline_config is not None:
                        metadata["pipeline_config"] = pipeline_config

                    # calc shift
                    cl_raw = np.squeeze(run_data_in["CL"]["detected"][:])
                    shift_y, shift_x = calculate_shift(cl_raw)

                    # construct sample
                    sample = {
                        "__key__": f"{stem}_{key}",
                        "json": json.dumps(metadata, default=str).encode("utf-8"),
                    }

                    for ds_name, (h5_path, is_mask) in datasets_to_process.items():
                        parts = h5_path.split("/")
                        node = run_data_in
                        exists = True
                        for p in parts:
                            if p in node:
                                node = node[p]
                            else:
                                exists = False
                                break

                        if exists:
                            raw_data = np.squeeze(node[:])
                            processed_data = transform_array(
                                raw_data,
                                shift_y,
                                shift_x,
                                crop_size=960,
                                target_size=224,
                                is_mask=is_mask,
                            )
                            sample[f"{ds_name}.tiff"] = array_to_tiff_bytes(
                                processed_data
                            )

                    sink.write(sample)

    print(f"Worker {worker_id} fertig. Shards gespeichert in '{reduced_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process H5 datasets into WebDataset shards."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="path to h5 data directory"
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="id of specific job in slurm array task",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of jobs in task array"
    )
    parser.add_argument(
        "--maxcount", type=int, default=1000, help="max sample count per shard-file"
    )

    args = parser.parse_args()
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(
            f"worker_id ({args.worker_id}) muss zwischen 0 und num_workers-1 ({args.num_workers-1}) liegen."
        )

    process_dataset(
        data_dir=args.data_dir,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        maxcount=args.maxcount,
    )
