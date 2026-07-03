import os
import logging
import mlflow
import hydra
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from hydra.utils import instantiate

from experiments.abstract_experiment import (
    AbstractExperiment,
    MLflowLoggingCallback,
    setup_mlflow_globals,
)
from datasets.abstract_dataset import AbstractDataset
from visualizations.segmentation_visualizations import visualize_segmentation_result

logger = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """Callback to visualize and log first batch during testing and validation"""

    def __init__(self, config: DictConfig, log_every_n_epochs: int = -1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.cfg = config

    @rank_zero_only
    def _log_visualization(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, filename: str
    ):
        if self.cfg.visualization_type == "multi_label_classification":
            return

        if self.cfg.visualization_type == "segmentation":
            batch_x, _, true_mask = batch
            pl_module.eval()
            with torch.no_grad():
                logits = pl_module(batch_x.to(pl_module.device))
                probs = torch.sigmoid(logits)
                predicted_mask = (probs > 0.5).float()

            fig = visualize_segmentation_result(
                batch_x[0, 0].cpu(), true_mask[0, 0].cpu(), predicted_mask[0, 0].cpu()
            )

            for logger in trainer.loggers:
                if isinstance(logger, MLFlowLogger) and fig is not None:
                    logger.experiment.log_figure(logger.run_id, fig, filename)
                    plt.close(fig)

            pl_module.train()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            self._log_visualization(
                trainer, pl_module, batch, "visualizations/test_reconstruction.png"
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (
            self.log_every_n_epochs != -1
            and batch_idx == 0
            and trainer.current_epoch % self.log_every_n_epochs == 0
        ):
            filename = f"visualizations/val_reconstruction_epoch_{trainer.current_epoch:03d}.png"
            self._log_visualization(trainer, pl_module, batch, filename)


class BaselineExperiment(AbstractExperiment):
    def __init__(
        self,
        dataloader: pl.LightningDataModule,
        experiment_config: DictConfig,
        config: DictConfig,
    ):
        super().__init__(
            name=experiment_config.name,
            dataloader=dataloader,
            checkpoint_dir=experiment_config.checkpoint_dir,
            config=config,
        )
        self.cfg = experiment_config
        self.model = instantiate(experiment_config.model)
        self.run_id = None

    def train_and_evaluate(self):
        mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.mlflow_uri,
            experiment_name="X-Ray Holography",
            run_name=self.name,
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

        mlflow_callback = MLflowLoggingCallback(
            config=self.config, experiment_name="X-Ray Holography"
        )

        vis_callback = VisualizationCallback(
            config=self.cfg, log_every_n_epochs=self.cfg.get("log_every_n_epochs", -1)
        )

        trainer = pl.Trainer(
            max_epochs=self.cfg.trainer.max_epochs,
            accelerator="auto",
            devices="auto",
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, mlflow_callback, vis_callback],
            precision="16-mixed",
        )
        self.run_id = trainer.logger.run_id

        logger.info("Starting downstream training...")
        trainer.fit(self.model, datamodule=self.dataloader)

        logger.info("Starting test evaluation...")
        trainer.test(self.model, datamodule=self.dataloader, ckpt_path="best")


@hydra.main(version_base=None, config_path="../../conf/", config_name="baseline_config")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.loglevel, format="%(levelname)s: %(message)s")
    setup_mlflow_globals(cfg)

    datamodule: AbstractDataset = instantiate(cfg.datamodule, batch_size=cfg.batch_size)
    datamodule.setup()

    experiment = cfg.models.baselines
    downstream_experiment = BaselineExperiment(
        dataloader=datamodule,
        experiment_config=experiment,
        config=cfg,
    )
    downstream_experiment.train_and_evaluate()


if __name__ == "__main__":
    main()
