## Docker Setup Guide

This guide exmplains how to setup a mlflow docker stack including, mlflow, an external database and a bucket storage system for saving artefacts.

### Setup Guide
A docker installation is requiered. Please follow the following steps:

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

### Adjust Experimental Configs
Each experiment config contains a `mlflow_uri` hydra-parameter. Please set this address to your now hosted mlflow instance, for instance `http://localhost:5000`.