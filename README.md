# Implementation of my Bachelors Thesis
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

HERE WILL BE A SHORT SUMMARY OF THE PROJECT [paper](https://uni-kassel.de).

## 🚀 Getting started
To replicate the experiments detailed in my thesis, please execute the following procedural steps:

### 1. Install Dependencies
Install all requisite software packages specified within the provided [requirements.txt](requirements.txt) file.

### 2. Docker Environment
Result tracking and metric logging are facilitated via MLflow. Therefore, an active MLflow instance is required. Researchers utilizing an existing instance must adjust the configuration paths accordingly. Alternatively, to deploy your own mlflow instance follow these steps:

1. Create an `.env` file in the [docker](/docker/) directory and paste the following configuration:
```bash
POSTGRES_USER=yourPostgresUser          
POSTGRES_PASSWORD=yourpostgresPassword 
MINIO_ROOT_USER=yourMinioUser              
MINIO_ROOT_PASSWORD=yourMinioPassword     
MINIO_ACCESS_KEY_ID=yourMinioUser
MINIO_SECRET_ACCESS_KEY=yourMinioPassword
```
⚠️ **Security Note**: For demonstration purposes, the `MINIO_ROOT_USER` password and the MinIO secret are identical. For secure deployments, particularly those accessible via external networks, robust and distinct credentials must be provisioned.

1. Initialize the Docker stack by executing `docker compose up -d` within the [docker](/docker/) directory
2. Following initialization, access the [web configuration](http://localhost:9001/login) using your designated `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` credentials to create a MinIO bucket named `mlflow`
3. Upon bucket creation, restart the Docker stack by sequentially executing `docker compose down` and `docker compose up -d`

### 3. Experiment configurations
Experimental parameters are systematically managed using Hydra. Template configuration files are located in the [conf](/conf/) directory. Researchers should modify these parameters as dictated by specific experimental or dataset requirements.

### 4. Setting Python Path
Ensure the environment path is set up correctly by executing `export PYTHONPATH=$PYTHONPATH:./` on Linux-based systems, or `$env:PYTHONPATH = "$env:PYTHONPATH;.\"` in Windows PowerShell within the [src](/src/) directory.

MAYBE MORE STEPS WILL BE NECESSARY

## 📂 Project structure
A SOURCE FILE TREE WILL BE SHOWN HERE IN THE END