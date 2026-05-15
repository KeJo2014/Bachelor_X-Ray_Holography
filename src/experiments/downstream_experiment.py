import os
import pytorch_lightning as pl
import logging
import mlflow
import hydra
import importlib
import torch
import matplotlib.pyplot as plt
import glob

from experiments.abstract_experiment import AbstractExperiment
from datasets.hologram_dataset import HologramDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig
from hydra.utils import instantiate, get_class

logger = logging.getLogger(__name__)


class DownstreamExperiment(AbstractExperiment):
    def __init__(
        self,
        dataloader: pl.LightningDataModule,
        mlflow_run_id: str,
        experiment_config: DictConfig,
        config: DictConfig,
    ):
        super().__init__(
            name=experiment_config.name,
            dataloader=dataloader,
            mlflow_run_id=mlflow_run_id,
            checkpoint_dir=os.path.join(experiment_config.checkpoint_dir),
            config=config,
        )
        self.cfg = experiment_config
        self.model = self._build_model()

    def _get_class_from_string(self, class_path: str):
        """Helper function to load python class according to path"""
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _build_model(self) -> pl.LightningModule:
        logger.info(
            f"Currently loading trained backbone from file: {self.cfg.backbone.checkpoint_path}"
        )
        backbone_class = self._get_class_from_string(self.cfg.backbone.module_class)
        pretrained_model = backbone_class.load_from_checkpoint(
            self.cfg.backbone.checkpoint_path
        )
        # extract encoder from model
        encoder = pretrained_model.encoder

        logger.info("Initializing new downstream head...")
        head = instantiate(self.cfg.task.head)

        # attach both
        logger.info("Creating task module...")
        task_module = instantiate(
            self.cfg.task,
            encoder=encoder,
            head=head,
            _recursive_=False,
        )

        return task_module

    def train_and_evaluate(self):
        mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.mlflow_uri,
            run_name=self.name,
            run_id=self.mlflow_run_id,
        )

        monitor_metric = self.cfg.trainer.monitor_metric
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=f"task-{{epoch:02d}}-{monitor_metric.replace('/', '_')}={{{monitor_metric}:.4f}}",
            monitor=monitor_metric,
            mode=self.cfg.trainer.monitor_mode,
            save_top_k=1,
            auto_insert_metric_name=False,
        )

        trainer = pl.Trainer(
            max_epochs=self.cfg.trainer.max_epochs,
            accelerator="auto",
            devices=1,
            logger=mlflow_logger,
            callbacks=[checkpoint_callback],
            precision="16-mixed",
        )

        logger.info("Starting downstream training...")
        trainer.fit(self.model, datamodule=self.dataloader)

        logger.info("Starting test evaluation...")
        trainer.test(self.model, datamodule=self.dataloader, ckpt_path="best")

    def create_segmentation_visualizations(self):
        from visualizations.segmentation_visualizations import (
            visualize_segmentation_result,
        )

        self._load_model_from_checkpoint(model_type=self.model.__class__)
        test_loader = self.dataloader.test_dataloader()
        batch_x, _, true_mask = next(iter(test_loader))

        x = batch_x.to(self.model.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)
            predicted_mask = (probs > 0.5).float()

        fig = visualize_segmentation_result(
            batch_x[0, 0], true_mask[0, 0], predicted_mask.cpu()[0, 0]
        )
        mlflow.log_figure(fig, "visualizations/segmentation_result.png")
        plt.close(fig)

    def _load_model_from_checkpoint(self, model_type: pl.LightningModule):
        """
        Loads newest checkpoint for provided model type.
        """
        search_path = os.path.join(self.checkpoint_dir, "*.ckpt")
        checkpoint_files = glob.glob(search_path)

        if not checkpoint_files:
            logger.critical(
                "No checkpoint file could be found for Random MAE model. Exiting."
            )
            raise FileNotFoundError(f"No model checkpoint found.")

        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        logger.info(f"Loading newst model checkpoint: {latest_checkpoint}")

        self.model = model_type.load_from_checkpoint(
            latest_checkpoint, encoder=self.model.encoder, head=self.model.head
        )


@hydra.main(
    version_base=None, config_path="../../conf/", config_name="downstream_config"
)
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.loglevel, format="%(levelname)s: %(message)s")

    mlflow.set_tracking_uri(uri=cfg.mlflow_uri)
    mlflow.set_experiment("X-Ray Holography")

    datamodule = HologramDataModule(
        data_dir=os.path.join(cfg.data_dir),
        batch_size=cfg.batch_size,
    )
    datamodule.setup()
    for experiment in cfg.experiments:
        with mlflow.start_run(run_name=experiment.name) as run:
            mlflow.log_param("freeze_encoder", experiment.task.freeze_encoder)

            downstream_experiment = DownstreamExperiment(
                dataloader=datamodule,
                mlflow_run_id=run.info.run_id,
                experiment_config=experiment,
                config=cfg,
            )
            downstream_experiment.train_and_evaluate()
            if experiment.visualization_type == "segmentation":
                downstream_experiment.create_segmentation_visualizations()
            elif experiment.visualization_type == "multi_label_classification":
                pass


if __name__ == "__main__":
    main()
