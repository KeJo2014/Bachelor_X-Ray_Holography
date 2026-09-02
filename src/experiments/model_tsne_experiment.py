import os
import pytorch_lightning as pl
import logging
import hydra
import importlib
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tempfile

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from metrics.clustering_metrics import get_metric_collection
from experiments.abstract_experiment import (
    AbstractExperiment,
    setup_mlflow_globals,
)
from datasets.abstract_dataset import AbstractDataset
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from visualizations.tsne_visualizations import (
    create_2d_tsne_plot,
    create_3d_tsne_visualization,
    create_2d_contour_plot,
    create_orthogonal_projections,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class FeatureEvalExperiment(AbstractExperiment):
    def __init__(
        self,
        dataloader: pl.LightningDataModule,
        experiment_config: DictConfig,
        config: DictConfig,
    ):
        super().__init__(
            name=experiment_config.name,
            dataloader=dataloader,
            checkpoint_dir=None,
            config=config,
        )
        self.cfg = experiment_config
        self.run_id = None

    def _get_class_from_string(self, class_path: str):
        """Helper function to load python class according to path"""
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _extracted_loaded_encoder(self) -> pl.LightningModule:
        """Helper function to load the pretrained encoder from the checkpoint or native DINOv3 model"""
        if self.cfg.get("use_pretrained_dino", True):
            import timm

            logger.info("Loading native dinov3 model with frozen weights")
            encoder = timm.create_model(
                "vit_base_patch16_dinov3.lvd1689m",
                pretrained=True,
                num_classes=0,
                in_chans=self.cfg.channels,
                global_pool="",
            )
            for param in encoder.parameters():
                param.requires_grad = False

        else:
            logger.info(
                f"Extrahiere Encoder-Gewichte aus Downstream-Checkpoint: {self.cfg.checkpoint_path}"
            )
            pretext_model = instantiate(self.cfg.encoder)
            encoder = pretext_model.encoder
            checkpoint = torch.load(self.cfg.checkpoint_path, map_location="cpu")

            # get encoder weights
            encoder_weights = {
                k.replace("encoder.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("encoder.")
            }
            encoder.load_state_dict(encoder_weights, strict=True)

        return encoder

    def eval_feature_embedding(self):
        mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.mlflow_uri,
            experiment_name="X-Ray Holography",
            run_name=f"{self.name}_eval",
        )

        if mlflow_logger.run_id:
            mlflow_logger.experiment.log_text(
                mlflow_logger.run_id,
                OmegaConf.to_yaml(self.config, resolve=True),
                "config.yaml",
            )

        encoder = self._extracted_loaded_encoder()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.to(device)
        encoder.eval()

        logger.info(f"Start embedding features using pretrained models encode")

        # calculate test feature embeddings
        embeddings = []
        labels = []

        test_dataloader = self.dataloader.test_dataloader()
        label_map = getattr(self.dataloader, "label_map", None)
        inv_map = {v: k for k, v in label_map.items()} if label_map else None

        with torch.no_grad():
            for batch in test_dataloader:
                x, y, masks = self.dataloader.on_after_batch_transfer(
                    batch, dataloader_idx=0
                )
                x = x.to(device)

                # extract features
                features = encoder(x)
                if features.ndim > 2:
                    features = (
                        features.mean(dim=[2, 3])
                        if len(features.shape) == 4
                        else features.mean(dim=1)
                    )

                embeddings.append(features.cpu())
                labels.append(y.cpu())

        embeddings = torch.cat(embeddings).numpy()
        labels = torch.cat(labels).numpy()

        if inv_map:
            str_labels = [inv_map.get(lbl, str(lbl)) for lbl in labels]
        else:
            str_labels = [str(lbl) for lbl in labels]

        logger.info("Computing 2D t-SNE embeddings for static plot...")
        tsne_2d = TSNE(n_components=2, random_state=42)
        reduced_embeddings_2d = tsne_2d.fit_transform(embeddings)

        # create static matplot 2d tsne visualization
        fig_static = create_2d_tsne_plot(reduced_embeddings_2d, str_labels)

        if mlflow_logger.run_id:
            mlflow_logger.experiment.log_figure(
                mlflow_logger.run_id,
                fig_static,
                "visualizations/tsne_embeddings_2d.pdf",
            )
        plt.close(fig_static)

        logger.info("Computing 2d t-SNE contour plots plot...")
        fig = create_2d_contour_plot(reduced_embeddings_2d, str_labels)
        if mlflow_logger.run_id:
            mlflow_logger.experiment.log_figure(
                mlflow_logger.run_id,
                fig,
                "visualizations/tsne_contour_2d.pdf",
            )
        plt.close(fig)

        logger.info("Computing 3D t-SNE embeddings for interactive plot...")
        tsne_3d = TSNE(n_components=3, random_state=42)
        reduced_embeddings_3d = tsne_3d.fit_transform(embeddings)
        fig_interactive = create_3d_tsne_visualization(
            reduced_embeddings_3d, str_labels
        )

        if mlflow_logger.run_id:
            with tempfile.TemporaryDirectory() as tmpdir:
                html_path = os.path.join(tmpdir, "tsne_embeddings_3d.html")
                fig_interactive.write_html(html_path)
                mlflow_logger.experiment.log_artifact(
                    mlflow_logger.run_id, html_path, artifact_path="visualizations"
                )

        logger.info("Computing 3 orthogonal 2D projections from 3D t-SNE embeddings...")
        fig = create_orthogonal_projections(reduced_embeddings_3d, str_labels)
        if mlflow_logger.run_id:
            mlflow_logger.experiment.log_figure(
                mlflow_logger.run_id,
                fig,
                "visualizations/tsne_3d_perspectives.pdf",
            )
        plt.close(fig)

        logger.info(f"t-SNE visualizations logged to MLFlow.")

        if len(np.unique(labels)) > 1:
            n_classes = len(np.unique(labels))
            logger.info(f"Running K-Means with k={n_classes} for calculation...")

            # execute k-means on embeddings
            kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
            cluster_preds = kmeans.fit_predict(embeddings)

            metrics = get_metric_collection()
            preds_tensor = torch.tensor(cluster_preds)
            labels_tensor = torch.tensor(labels)
            metric_results = metrics(preds_tensor, labels_tensor)

            ari_score = metric_results["adjusted_rand_score"].item()
            nmi_score = metric_results["normalized_mutual_info_score"].item()

            logger.info(
                f"K-Means Clustering - ARI: {ari_score:.4f} | NMI: {nmi_score:.4f}"
            )

            if mlflow_logger.run_id:
                mlflow_logger.experiment.log_metric(
                    mlflow_logger.run_id, "ari_score", ari_score
                )
                mlflow_logger.experiment.log_metric(
                    mlflow_logger.run_id, "nmi_score", nmi_score
                )
                if mlflow_logger.run_id:
                    mlflow_logger.experiment.set_terminated(
                        mlflow_logger.run_id, status="FINISHED"
                    )


@hydra.main(version_base=None, config_path="../../conf/", config_name="tsne_config")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.loglevel, format="%(levelname)s: %(message)s")
    setup_mlflow_globals(cfg)

    datamodule: AbstractDataset = instantiate(cfg.datamodule, batch_size=cfg.batch_size)
    datamodule.setup()
    experiment = cfg.model

    feature_evaluation_experiment = FeatureEvalExperiment(
        dataloader=datamodule,
        experiment_config=experiment,
        config=cfg,
    )
    feature_evaluation_experiment.eval_feature_embedding()


if __name__ == "__main__":
    main()
