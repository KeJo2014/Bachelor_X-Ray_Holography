# Decoding Diffraction <br>Self-Supervised Architectures for 2D-Fourier Data
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This research explores the application of Self-Supervised Learning (SSL) frameworks to process 2D Fourier data in X-ray holography. It demonstrates that complex downstream tasks can be effectively resolved using robust feature representations learned by adopting the Masked Autoencoder framework to the constraints of X-ray holography. For a comprehensive analysis, please refer to my [thesis](./misc/media/thesis.pdf).

## 🚀 Getting started
To replicate the experiments detailed in this thesis, please execute the following procedural steps:

### 1. Install Dependencies
Install all requisite software packages specified within the provided [requirements.txt](requirements.txt) file.

In this work, we employ mlflow to track our experimental results. There are multiple options to run this on your own machine. The easiest solution is to leave the current configuration as is. This will create a local mlflow instance on your machine using an sqlite database. If you want to run experiments in parallel, we strongly advise to setup an mlflow stack as described in the following [guide](misc/documentation/docker_setup.md).

### 2. Experiment configurations
Experimental parameters are systematically managed using Hydra. Template configuration files are located in the [conf](/conf/) directory. Researchers should modify these parameters as dictated by specific experimental or dataset requirements.

### 3. Setting Python Path
Ensure the environment path is set up correctly by executing `export PYTHONPATH=$PYTHONPATH:./` on Linux-based systems, or `$env:PYTHONPATH = "$env:PYTHONPATH;.\"` in Windows PowerShell within the [src](/src/) directory.

## 📂 Project structure
```tree
.
├── README.md
├── conf
│   ├── backbone_config.yaml
│   ├── baseline_config.yaml
│   ├── downstream_config.yaml
│   ├── models
│   │   ├── backbones
│   │   │   ├── dinov3.yaml
│   │   │   ├── mae.yaml
│   │   │   └── sim_mim.yaml
│   │   ├── baselines
│   │   │   ├── resnet_18.yaml
│   │   │   └── unet.yaml
│   │   └── downstream
│   │       ├── classification.yaml
│   │       └── segmentation_dpt.yaml
│   └── tsne_config.yaml
├── docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── src
    ├── datasets
    │   ├── abstract_dataset.py
    │   ├── reduce_dataset.py
    │   └── simulated_dataset.py
    ├── experiments
    │   ├── abstract_experiment.py
    │   ├── backbone_experiment.py
    │   ├── baseline_experiment.py
    │   ├── downstream_experiment.py
    │   └── model_tsne_experiment.py
    ├── metrics
    │   ├── classification_metrics.py
    │   ├── clustering_metrics.py
    │   └── segmentation_metrics.py
    ├── models
    │   ├── baselines
    │   │   ├── resnet18_baseline.py
    │   │   └── unet_baseline.py
    │   ├── heads
    │   │   ├── classification_head.py
    │   │   └── segmentation_dpt_head.py
    │   ├── lightning_modules
    │   │   ├── backbones
    │   │   │   ├── dinov3.py
    │   │   │   ├── mae.py
    │   │   │   └── sim_mim.py
    │   │   ├── pretext_tasks
    │   │   │   ├── __init__.py
    │   │   │   ├── pretext_task_action.py
    │   │   │   ├── random_masking.py
    │   │   │   └── random_real_space_masking.py
    │   │   └── tasks
    │   │       ├── classification_task.py
    │   │       └── segmentation_task.py
    │   └── loss_functions
    │       ├── __init__.py
    │       ├── center_focused_tversky_loss.py
    │       ├── dice_loss.py
    │       └── radially_weighted_loss.py
    └── visualizations
        ├── backbone_visualizations.py
        ├── dataset_visualizations.py
        ├── segmentation_visualizations.py
        └── tsne_visualizations.py
```