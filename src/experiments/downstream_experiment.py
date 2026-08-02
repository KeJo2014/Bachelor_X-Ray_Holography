import os
import pytorch_lightning as pl
import logging
import hydra
import importlib
import torch
import matplotlib
import matplotlib.pyplot as plt
import heapq

from experiments.abstract_experiment import (
    AbstractExperiment,
    MLflowLoggingCallback,
    setup_mlflow_globals,
)
from datasets.abstract_dataset import AbstractDataset
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from visualizations.segmentation_visualizations import visualize_segmentation_result
from visualizations.backbone_visualizations import plot_multiclass_confusion_matrix
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchmetrics.functional.classification import binary_fbeta_score

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """Callback to visualize and log first batch and worst predictions during testing."""

    def __init__(self, config, log_every_n_epochs: int = -1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.cfg = config

        # priority queue for bad predictions
        self.worst_predictions = []
        self.counter = 0

    @rank_zero_only
    def _log_visualization(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, filename: str
    ):
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

        if self.cfg.visualization_type != "segmentation":
            return

        batch_x, _, true_mask = batch
        predicted_mask = outputs["preds"]

        batch_size = batch_x.size(0)
        target_mask = true_mask.long().to(pl_module.device)

        for b in range(batch_size):
            f2 = binary_fbeta_score(
                predicted_mask[b].flatten(), target_mask[b].flatten(), beta=2.0
            ).item()

            item = (
                -f2,
                self.counter,
                batch_x[b, 0].cpu(),
                true_mask[b, 0].cpu(),
                predicted_mask[b, 0].cpu(),
            )
            self.counter += 1

            if len(self.worst_predictions) < 20:
                heapq.heappush(self.worst_predictions, item)
            else:
                heapq.heappushpop(self.worst_predictions, item)

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

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        if self.cfg.visualization_type == "multi_class_classification":
            matrix = pl_module.test_conf_mat.compute()

            # create class_name list
            label_map = trainer.datamodule.label_map
            inv_map = {v: k for k, v in label_map.items()}
            class_names = [inv_map[i] for i in range(pl_module.hparams.num_classes)]

            fig = plot_multiclass_confusion_matrix(
                matrix=matrix,
                class_names=class_names,
            )

            for logger in trainer.loggers:
                if isinstance(logger, MLFlowLogger):
                    logger.experiment.log_figure(
                        logger.run_id,
                        fig,
                        "visualizations/multilabel_confusion_matrix.png",
                    )
            plt.close(fig)
            pl_module.test_conf_mat.reset()
        elif self.cfg.visualization_type == "segmentation":
            if not self.worst_predictions:
                return

            self.worst_predictions.sort(key=lambda x: x[0], reverse=True)

            for i, (neg_f2, _, img_cpu, true_m_cpu, pred_m_cpu) in enumerate(
                self.worst_predictions
            ):
                f2_val = -neg_f2
                fig = visualize_segmentation_result(
                    img_cpu, true_m_cpu, pred_m_cpu.float()
                )

                for logger in trainer.loggers:
                    if (
                        hasattr(logger, "experiment")
                        and hasattr(logger.experiment, "log_figure")
                        and fig is not None
                    ):
                        logger.experiment.log_figure(
                            logger.run_id,
                            fig,
                            f"visualizations/error_masks/error_rank_{i}_f2_{f2_val:.3f}.png",
                        )
                plt.close(fig)
            self.worst_predictions.clear()
            self.counter = 0


class DownstreamExperiment(AbstractExperiment):
    def __init__(
        self,
        dataloader: pl.LightningDataModule,
        experiment_config: DictConfig,
        config: DictConfig,
    ):
        super().__init__(
            name=experiment_config.name,
            dataloader=dataloader,
            checkpoint_dir=os.path.join(experiment_config.checkpoint_dir),
            config=config,
        )
        self.cfg = experiment_config
        self.model = self._build_model()
        self.run_id = None

    def _get_class_from_string(self, class_path: str):
        """Helper function to load python class according to path"""
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _build_model(self) -> pl.LightningModule:
        if self.cfg.backbone.get("use_pretrained_dino", True):
            import timm

            logger.info("Loading native dinov3 model with frozen weights")
            encoder = timm.create_model(
                "vit_base_patch16_dinov3.lvd1689m",
                pretrained=True,
                num_classes=0,
                in_chans=self.cfg.backbone.channels,
                global_pool="",
            )

            for param in encoder.parameters():
                param.requires_grad = False

            encoder.eval()
        else:
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

        if hasattr(head, "inject_pretrained_encoder"):
            head.inject_pretrained_encoder(encoder)

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
            precision="bf16-mixed",
            gradient_clip_val=1.0,
        )
        self.run_id = trainer.logger.run_id

        logger.info("Starting downstream training...")
        trainer.fit(self.model, datamodule=self.dataloader)

        logger.info("Starting test evaluation...")
        trainer.test(self.model, datamodule=self.dataloader, ckpt_path="best")

    def evaluate_only(self):
        mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.mlflow_uri,
            experiment_name="X-Ray Holography",
            run_name=f"{self.name}_eval",
        )
        mlflow_callback = MLflowLoggingCallback(
            config=self.config, experiment_name="X-Ray Holography"
        )

        vis_callback = VisualizationCallback(
            config=self.cfg, log_every_n_epochs=self.cfg.get("log_every_n_epochs", -1)
        )

        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=mlflow_logger,
            callbacks=[vis_callback, mlflow_callback],
            precision="16-mixed",
        )
        ckpt_path = self.cfg.get("checkpoint_path")
        if not ckpt_path:
            raise ValueError(
                "For evaluation only puropses must 'checkpoint_path' in conf be set."
            )

        logger.info(f"Starting test evaluation mit Checkpoint: {ckpt_path}")
        trainer.test(self.model, datamodule=self.dataloader, ckpt_path=ckpt_path)


@hydra.main(
    version_base=None, config_path="../../conf/", config_name="downstream_config"
)
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.loglevel, format="%(levelname)s: %(message)s")
    setup_mlflow_globals(cfg)

    datamodule: AbstractDataset = instantiate(cfg.datamodule, batch_size=cfg.batch_size)
    datamodule.setup()
    experiment = cfg.models.downstream

    downstream_experiment = DownstreamExperiment(
        dataloader=datamodule,
        experiment_config=experiment,
        config=cfg,
    )
    if not bool(cfg.eval_only_mode):
        downstream_experiment.train_and_evaluate()
    else:
        downstream_experiment.evaluate_only()


if __name__ == "__main__":
    main()
