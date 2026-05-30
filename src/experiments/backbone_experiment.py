import os
import pytorch_lightning as pl
import logging
import mlflow
import matplotlib.pyplot as plt
import hydra

from experiments.abstract_experiment import AbstractExperiment
from datasets.abstract_dataset import AbstractDataset
from visualizations.mae_visualizations import visualize_mae_results
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig
from hydra.utils import instantiate, get_class


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
            callbacks=[checkpoint_callback],
            precision="16-mixed",  # use half-precision
        )
        trainer.fit(model, datamodule=self.dataloader)
        self.model = model

    def evaluate_model(self, model_type: pl.LightningModule):
        if self.model == None:
            self._load_model_from_checkpoint(model_type=model_type)

        mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.mlflow_uri,
            run_name="Evaluation",
            run_id=self.mlflow_run_id,
        )

        trainer = pl.Trainer(accelerator="auto", devices=1, logger=mlflow_logger)
        trainer.test(self.model, datamodule=self.dataloader)

    def generate_evaluation_visualization(self, model_type: pl.LightningModule):
        if self.model == None:
            self._load_model_from_checkpoint(model_type=model_type)

        test_loader = self.dataloader.test_dataloader()
        batch_x, _, _ = next(iter(test_loader))
        fig = visualize_mae_results(self.model, batch_x)
        mlflow.log_figure(fig, "visualizations/mae_reconstruction.png")
        plt.close(fig)


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
        model = instantiate(variation.parameters.model, img_size=datamodule.img_size)
        ModelClass = get_class(variation.parameters.model._target_)
        experiment.train_model(model)

        # get validation loss for the optuna optimizer
        current_val_loss = experiment.model.trainer.callback_metrics.get("val/loss")
        if current_val_loss is not None:
            best_val_loss = min(best_val_loss, current_val_loss.item())

        if not cfg.get("hyperparameter_optimization_mode", False):
            experiment.evaluate_model(ModelClass)
            experiment.generate_evaluation_visualization(ModelClass)

    return best_val_loss


if __name__ == "__main__":
    main()
