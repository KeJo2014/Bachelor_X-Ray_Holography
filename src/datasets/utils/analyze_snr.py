import h5py
import torch
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyze_dataset_crop_size(h5_filepath, sigma_multiplier=5.0, corner_size=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for matrix math: {device}")

    crop_sizes = []

    with h5py.File(h5_filepath, "r") as f:
        run_keys = [k for k in f.keys() if k != "_pipeline_config"]

        for key in tqdm(run_keys, desc="Analyzing SNR & Radii"):
            for side in ["CL", "CR"]:
                holo = np.squeeze(f[key][side]["detected"][:])

                # calculate holo center
                threshold = np.percentile(holo, 99.0)
                bright_core = holo > threshold
                cy_np, cx_np = scipy.ndimage.center_of_mass(bright_core)

                holo_t = torch.tensor(holo, device=device, dtype=torch.float32)
                h, w = holo_t.shape

                # estimate background noise
                corners = torch.cat(
                    [
                        holo_t[:corner_size, :corner_size].flatten(),
                        holo_t[:corner_size, -corner_size:].flatten(),
                        holo_t[-corner_size:, :corner_size].flatten(),
                        holo_t[-corner_size:, -corner_size:].flatten(),
                    ]
                )
                noise_mean = corners.mean()
                noise_std = corners.std()

                # define signal threshold
                threshold = noise_mean + (sigma_multiplier * noise_std)
                signal_mask = holo_t > threshold

                if not signal_mask.any():
                    continue

                # coordinate grid
                y_coords = torch.arange(h, device=device, dtype=torch.float32) - cy_np
                x_coords = torch.arange(w, device=device, dtype=torch.float32) - cx_np
                Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

                # distance matrix
                R = torch.sqrt(X**2 + Y**2)
                valid_radii = R[signal_mask]
                max_radius = torch.quantile(valid_radii, 0.999).item()

                # required crop size equal to diamter
                required_crop = int(np.ceil(max_radius * 2))
                if required_crop % 2 != 0:
                    required_crop += 1

                crop_sizes.append(required_crop)

    crop_sizes = np.array(crop_sizes)
    global_min = crop_sizes.min()
    global_max = crop_sizes.max()
    global_avg = crop_sizes.mean()

    print("\n--- SNR Crop Analysis Results ---")
    print(f"Total Holograms Processed: {len(crop_sizes)}")
    print(f"Global MIN Crop: {global_min} px")
    print(f"Global MAX Crop: {global_max} px")
    print(f"Global AVG Crop: {global_avg:.2f} px")
    print(f"99th Percentile: {np.percentile(crop_sizes, 99):.0f} px")

    plt.hist(crop_sizes, bins=50, color="teal", edgecolor="black")
    plt.title("Distribution of Required Crop Sizes (SNR based)")
    plt.xlabel("Required Crop Size [Pixels]")
    plt.ylabel("Frequency")
    plt.axvline(
        global_max,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Max: {global_max}",
    )
    plt.legend()
    plt.show()

    return global_min, global_max, global_avg


if __name__ == "__main__":
    h5_path = (
        "C:\\Users\\kelle\\Documents\\storage\\xray\\Raw_holo_sim\\master_dataset.h5"
    )
    analyze_dataset_crop_size(h5_path)
