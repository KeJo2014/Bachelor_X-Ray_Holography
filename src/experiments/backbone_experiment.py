import os
import pytorch_lightning as pl
import logging
import matplotlib.pyplot as plt
import hydra

from experiments.abstract_experiment import (
    AbstractExperiment,
    MLflowLoggingCallback,
    setup_mlflow_globals,
)
from datasets.abstract_dataset import AbstractDataset
from visualizations.backbone_visualizations import visualize_backbone_results
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig
from hydra.utils import instantiate, get_class


class MAEVisualizationCallback(Callback):
    """Callback to visualize and log first batch during testing and validation"""

    def __init__(self, log_every_n_epochs: int = -1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    @rank_zero_only
    def _log_visualization(self, trainer, pl_module, batch, filename):
        batch_x, _, _ = batch
        fig = visualize_backbone_results(pl_module, batch_x)

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(logger.run_id, fig, filename)
        plt.close(fig)

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


class RandomSimMIMExperiment(AbstractExperiment):
    def __init__(
        self,
        checkpoint_dir: os.PathLike,
        dataloader: pl.LightningDataModule,
        config: DictConfig,
        model_settings: DictConfig,
    ):
        """
        Experiment class that can train and evaluate a backbone selfSL model.
        """
        super().__init__(
            name=model_settings.name,
            dataloader=dataloader,
            checkpoint_dir=checkpoint_dir,
            config=config,
        )
        self.model_settings = model_settings
        self.model = None
        self.mlflow_logger = None

    def train_model(self, model: pl.LightningModule):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=f"backbone-{{epoch:02d}}-val_loss={{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        )

        vis_callback = MAEVisualizationCallback(
            log_every_n_epochs=self.model_settings.parameters.get(
                "log_every_n_epochs", -1
            )
        )
        self.mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.mlflow_uri,
            experiment_name="X-Ray Holography",
            run_name=self.name,
        )
        mlflow_callback = MLflowLoggingCallback(
            config=self.config, experiment_name="X-Ray Holography"
        )

        trainer = pl.Trainer(
            max_epochs=self.model_settings.parameters.num_epochs,
            accelerator="auto",
            devices="auto",
            logger=self.mlflow_logger,
            callbacks=[checkpoint_callback, vis_callback, mlflow_callback],
            precision="16-mixed",
            accumulate_grad_batches=2,
        )
        trainer.fit(model, datamodule=self.dataloader)
        self.model = model

    def evaluate_model(self, model_type: pl.LightningModule):
        if self.model == None:
            self._load_model_from_checkpoint(model_type=model_type)

        vis_callback = MAEVisualizationCallback()

        if self.mlflow_logger == None:
            self.mlflow_logger = MLFlowLogger(
                tracking_uri=self.config.mlflow_uri,
                experiment_name="X-Ray Holography",
                run_name=self.name,
            )

        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=self.mlflow_logger,
            callbacks=[vis_callback],
            accumulate_grad_batches=2,
        )
        trainer.test(self.model, datamodule=self.dataloader)


@hydra.main(version_base=None, config_path="../../conf/", config_name="backbone_config")
def main(cfg: DictConfig):
    # setup logging
    logging.basicConfig(level=cfg.loglevel, format="%(levelname)s: %(message)s")
    setup_mlflow_globals(cfg)
    logging.info("Initializing Processes for Random MAE Experiment...")

    datamodule: AbstractDataset = instantiate(
        cfg.datamodule, batch_size=cfg.experiments.batch_size
    )
    datamodule.setup()
    best_val_loss = float("inf")
    variation = cfg.models.backbones

    experiment = RandomSimMIMExperiment(
        checkpoint_dir=os.path.join(variation.checkpoint_dir),
        dataloader=datamodule,
        config=cfg,
        model_settings=variation,
    )
    ModelClass = get_class(variation.parameters.model._target_)

    if not cfg.eval_only_mode:
        model = instantiate(
            variation.parameters.model,
            img_size=variation.parameters.model.pretext_strategy.img_size,
        )
        experiment.train_model(model)

        current_val_loss = experiment.model.trainer.callback_metrics.get("val/loss")
        if current_val_loss is not None:
            best_val_loss = min(best_val_loss, current_val_loss.item())

    experiment.evaluate_model(ModelClass)
    return best_val_loss


if __name__ == "__main__":
    main()
