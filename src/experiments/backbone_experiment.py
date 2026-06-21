import os
import pytorch_lightning as pl
import logging
import mlflow
import matplotlib.pyplot as plt
import hydra

from experiments.abstract_experiment import AbstractExperiment
from datasets.abstract_dataset import AbstractDataset
from visualizations.mae_visualizations import visualize_mae_results
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig
from hydra.utils import instantiate, get_class


class MAEVisualizationCallback(Callback):
    """Callback to visualize and log first batch during testing and validation"""

    def __init__(self, log_every_n_epochs: int = -1):
        print(log_every_n_epochs)
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def _log_visualization(self, trainer, pl_module, batch, filename):
        batch_x, _, _ = batch
        fig = visualize_mae_results(pl_module, batch_x)

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(logger.run_id, fig, filename)
        plt.close(fig)
        print("logged: " + filename)

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
        mlflow_run_id: str,
        config: DictConfig,
        model_settings: DictConfig,
    ):
        """
        Experiment class that can train and evaluate a backbone selfSL model.
        """
        super().__init__(
            name=model_settings.name,
            dataloader=dataloader,
            mlflow_run_id=mlflow_run_id,
            checkpoint_dir=checkpoint_dir,
            config=config,
        )
        self.model_settings = model_settings
        self.model = None

    def train_model(self, model: pl.LightningModule):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="mae-{epoch:02d}-{train_loss:.4f}",
            monitor="train/loss_epoch",
            mode="min",
            save_top_k=3,
        )

        vis_callback = MAEVisualizationCallback(
            log_every_n_epochs=self.model_settings.parameters.get(
                "log_every_n_epochs", -1
            )
        )

        mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.mlflow_uri,
            run_name="Training",
            run_id=self.mlflow_run_id,
        )

        trainer = pl.Trainer(
            max_epochs=self.model_settings.parameters.num_epochs,
            accelerator="auto",
            devices=1,
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, vis_callback],
            precision="16-mixed",  # use half-precision
            accumulate_grad_batches=8,
        )
        trainer.fit(model, datamodule=self.dataloader)
        self.model = model

    def evaluate_model(self, model_type: pl.LightningModule):
        if self.model == None:
            self._load_model_from_checkpoint(model_type=model_type)

        vis_callback = MAEVisualizationCallback()

        mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.mlflow_uri,
            run_name="Evaluation",
            run_id=self.mlflow_run_id,
        )

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            logger=mlflow_logger,
            callbacks=[vis_callback],
        )
        trainer.test(self.model, datamodule=self.dataloader)


@hydra.main(version_base=None, config_path="../../conf/", config_name="backbone_config")
def main(cfg: DictConfig):
    # setup logging
    logging.basicConfig(level=cfg.loglevel, format="%(levelname)s: %(message)s")
    logging.info("Initializing Processes for Random MAE Experiment...")

    # setup mlflow
    mlflow.set_tracking_uri(uri=cfg.mlflow_uri)
    mlflow.set_experiment("X-Ray Holography")
    if cfg.get("mlflow_log_system_metrics", False):
        mlflow.enable_system_metrics_logging()

    datamodule: AbstractDataset = instantiate(
        cfg.datamodule, batch_size=cfg.experiments.batch_size
    )
    datamodule.setup()
    best_val_loss = float("inf")
    variation = cfg.models.backbones
    with mlflow.start_run(run_name=variation.name) as parent_run:
        mlflow.log_params(variation.parameters)
        experiment = RandomSimMIMExperiment(
            checkpoint_dir=os.path.join(variation.checkpoint_dir),
            dataloader=datamodule,
            mlflow_run_id=parent_run.info.run_id,
            config=cfg,
            model_settings=variation,
        )
        ModelClass = get_class(variation.parameters.model._target_)

        if not cfg.eval_only_mode:
            model = instantiate(
                variation.parameters.model, img_size=datamodule.img_size
            )
            experiment.train_model(model)
            # get validation loss for the optuna optimizer
            current_val_loss = experiment.model.trainer.callback_metrics.get("val/loss")
            if current_val_loss is not None:
                best_val_loss = min(best_val_loss, current_val_loss.item())

        if not cfg.get("hyperparameter_optimization_mode", False):
            experiment.evaluate_model(ModelClass)

    return best_val_loss


if __name__ == "__main__":
    main()
