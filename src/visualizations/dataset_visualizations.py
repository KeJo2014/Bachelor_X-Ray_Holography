import os
import glob
import logging
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import ast
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# CONSTANTS
DATA_DIR = Path(r"C:/Users/kelle/Documents/storage/xray/Raw_holo_sim")
VISUALIZATION_FILE_DESTINATION = (
    r"C:/Users/kelle/Documents/storage/xray/dataset_parameter_dist.png"
)
CSV_CACHE_DESTINATION = (
    r"C:/Users/kelle/Documents/storage/xray/dataset_metrics_cache.csv"
)
STATS_DESTINATION = r"C:/Users/kelle/Documents/storage/xray/dataset_statistics.csv"

FEATURES_TO_PLOT = [
    "pattern_type",
    "xray_energy",
    "detector_distance",
    "detector_center",
    "beamstop_center",
    "beamstop_angle",
    "number_frames",
    "illumination_fwhm",
    "counts_per_photon",
    "illumination_focus_distance",
    "max_counts_per_image",
    "exposure_time",
]

H5_PATH_MAPPING = {
    "pattern_type": "metadata/sample/magnetic_pattern/pattern_type_method",
    "xray_energy": "metadata/xray/energy_eV",
    "detector_distance": "metadata/detector/sample_to_detector_distance",
    "number_frames": "metadata/measurement_config/number_frames",
    "illumination_fwhm": "metadata/illumination/fwhm_m",
    "counts_per_photon": "metadata/detector_params/counts_per_photon",
    "exposure_time": "metadata/measurement_config/exposure_time",
    "sigma_photon": "metadata/artifacts_config/sigma_photon",
    "detector_center": "metadata/detector/detector_center",
    "illumination_focus_distance": "metadata/illumination/focus_distance_m",
    "max_counts_per_image": "metadata/measurement_config/max_counts_per_image",
    "beamstop_center": "metadata/beamstop/bs_center",
    "beamstop_radius": "metadata/beamstop/bs_config/radius",
    "beamstop_angle": "metadata/beamstop/bs_config/angle",
}


def get_h5_value(group, path):
    """Helper function to extract value from hdf5 path"""
    val = group[path][()]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray) and val.ndim == 0:
        return val.item()
    return val


def extract_or_load_data():
    """Check if cache is existent, else load hdf5 files"""
    if os.path.exists(CSV_CACHE_DESTINATION):
        logging.info(f"Loading cached statitics in {CSV_CACHE_DESTINATION}")
        return pd.read_csv(CSV_CACHE_DESTINATION)

    logging.info(f"No cache found. Traversing hdf5 files {DATA_DIR}...")
    h5_files = sorted(glob.glob(str(DATA_DIR / "*.h5")))

    if not h5_files:
        raise FileNotFoundError(f"No hdf5 files in {DATA_DIR} found")

    extracted_data = []

    for file_path in tqdm(h5_files, desc="Extracting hdf5 metadata..."):
        try:
            with h5py.File(file_path, "r") as f:
                run_keys = [
                    k for k in f.keys() if k != "_pipeline_config" and k.isdigit()
                ]

                for key in run_keys:
                    run_group = f[key]
                    row_data = {"file_name": os.path.basename(file_path), "run_id": key}

                    for feature in FEATURES_TO_PLOT:
                        if feature in H5_PATH_MAPPING:
                            h5_path = H5_PATH_MAPPING[feature]
                            val = get_h5_value(run_group, h5_path)

                            if isinstance(val, str) and (
                                val.startswith("(") or val.startswith("[")
                            ):
                                try:
                                    val = ast.literal_eval(val)
                                except (ValueError, SyntaxError):
                                    pass

                            if (
                                isinstance(val, (np.ndarray, list, tuple))
                                and len(val) == 2
                            ):
                                row_data[f"{feature}_y"] = float(val[0])
                                row_data[f"{feature}_x"] = float(val[1])
                            else:
                                row_data[feature] = val
                        else:
                            logging.warning(
                                f"Feature '{feature}' not defined in mapping."
                            )

                    extracted_data.append(row_data)
        except Exception as e:
            logging.error(f"Error while reading {file_path}: {e}")

    df = pd.DataFrame(extracted_data)

    # calc and store statistics
    logging.info(f"Saving metrics & statistics...")
    df.to_csv(CSV_CACHE_DESTINATION, index=False)
    df.describe(include="all").to_csv(STATS_DESTINATION)

    return df


df = extract_or_load_data()

logging.info(f"Erfolgreich {len(df)} Instanzen geladen.")
if "pattern_type" in df.columns:
    logging.info(f"Gefundene Pattern: {df['pattern_type'].dropna().unique().tolist()}")

# VISUALIZATION
sns.set_theme(style="whitegrid")
num_features = len(FEATURES_TO_PLOT)
cols = 3
rows = math.ceil(num_features / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
fig.suptitle(
    "Overview Simulated Parameter Distributions", fontsize=16, fontweight="bold"
)

for i, feature in enumerate(FEATURES_TO_PLOT):
    ax = axes[i]

    if f"{feature}_x" in df.columns and f"{feature}_y" in df.columns:
        sns.scatterplot(
            data=df,
            x=f"{feature}_x",
            y=f"{feature}_y",
            ax=ax,
            alpha=0.6,
            color="purple",
        )
        ax.set_title(f"2D Distribution: {feature}", fontsize=12)
        ax.set_xlabel(f"{feature} X")
        ax.set_ylabel(f"{feature} Y")

    # visualize categorical data
    elif df[feature].dtype == "object" or isinstance(
        df[feature].dropna().iloc[0] if not df[feature].dropna().empty else None, str
    ):
        sns.countplot(
            data=df, x=feature, ax=ax, hue=feature, palette="viridis", legend=False
        )
        ax.set_title(f"Distribution: {feature}", fontsize=12)
        ax.tick_params(axis="x", rotation=45)

    # visualize numerical data
    else:
        sns.histplot(data=df, x=feature, kde=True, ax=ax, color="steelblue", bins=20)
        ax.set_title(f"Distribution: {feature}", fontsize=12)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(VISUALIZATION_FILE_DESTINATION)
logging.info(f"visual saved: {VISUALIZATION_FILE_DESTINATION}")
