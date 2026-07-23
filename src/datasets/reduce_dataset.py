import glob
import h5py
import torch
import numpy as np
import scipy.ndimage
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm


def calculate_shift(image_np):
    """Calculate hologra center shift."""
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


def process_dataset(data_dir: str):
    data_dir = Path(data_dir)
    reduced_dir = data_dir / "reduced"
    reduced_dir.mkdir(exist_ok=True)

    h5_files = sorted(glob.glob(str(data_dir / "*.h5")))

    for file_path in h5_files:
        filename = Path(file_path).name
        out_path = reduced_dir / filename

        print(f"Process: {filename} -> {out_path}")

        with h5py.File(file_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
            if "_pipeline_config" in f_in.keys():
                f_in.copy("_pipeline_config", f_out)

            run_keys = [
                k for k in f_in.keys() if k != "_pipeline_config" and k.isdigit()
            ]

            for key in tqdm(run_keys, desc=f"Runs in {filename}", leave=False):
                run_group_out = f_out.create_group(key)

                # copy metadata
                if "metadata" in f_in[key]:
                    f_in.copy(f"{key}/metadata", run_group_out)
                if "sample" in f_in[key]:
                    f_in.copy(f"{key}/sample", run_group_out)

                run_data_in = f_in[key]

                # calc shift
                cl_raw = np.squeeze(run_data_in["CL"]["detected"][:])
                shift_y, shift_x = calculate_shift(cl_raw)

                datasets_to_process = {
                    "CL": ("CL/detected", False),
                    "CR": ("CR/detected", False),
                    "CL_no_beamstop": ("CL/detected_no_beamstop", False),
                    "CR_no_beamstop": ("CR/detected_no_beamstop", False),
                    "beamstop_mask": ("beamstop_mask", True),
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

                        if "/" in h5_path:
                            group_name, ds_name = h5_path.split("/")
                            if group_name not in run_group_out:
                                run_group_out.create_group(group_name)
                            target_group = run_group_out[group_name]
                        else:
                            target_group = run_group_out
                            ds_name = h5_path

                        target_group.create_dataset(
                            ds_name,
                            data=processed_data,
                            dtype=np.float32,
                            chunks=True,
                            compression="lzf",
                        )


if __name__ == "__main__":
    data_path = "/home/jkeller/dataset"
    process_dataset(data_path)
    print("Done. File are stored in subfolder 'reduced'.")
