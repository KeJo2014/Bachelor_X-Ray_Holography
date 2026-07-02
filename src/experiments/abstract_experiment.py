import mlflow
import pytorch_lightning as pl
import torch
import logging
import os
import glob

from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
from models.lightning_modules.pretext_tasks import PRETEXT_STRATEGIES
from models.loss_functions import LOSS_FUNCTIONS
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals(PRETEXT_STRATEGIES)
torch.serialization.add_safe_globals(LOSS_FUNCTIONS)


@rank_zero_only
def setup_mlflow_globals(cfg):
    mlflow.set_tracking_uri(uri=cfg.mlflow_uri)
    if cfg.get("mlflow_log_system_metrics", False):
        mlflow.enable_system_metrics_logging()


class AbstractExperiment(ABC):
    """
    This is the abstract base class for experiments.
    """

    def __init__(
        self,
        name: str,
        dataloader: pl.LightningDataModule,
        checkpoint_dir: os.PathLike,
        config: DictConfig,
    ) -> None:
        """
        Initializes the Experiment.

        :param name: Name of the experiment.
        :return: None
        """
        super().__init__()
        self.dataloader: pl.LightningDataModule = dataloader
        self.name = name
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # create reproducibility
        pl.seed_everything(42)

        # check current system settings
        if torch.cuda.is_available():
            major_version, minor_version = torch.cuda.get_device_capability()
            # check if tensor cores are available
            if major_version >= 7:
                torch.set_float32_matmul_precision("medium")
                logger.info(
                    f"Tensor cores detected (Compute Capability {major_version}.{minor_version})."
                )
            else:
                logging.info(
                    f"GPU detected (Compute Capability {major_version}.{minor_version}), no tensor cores available."
                )
        else:
            logging.info("No GPU detected.")

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

        self.model = model_type.load_from_checkpoint(latest_checkpoint)

    # @abstractmethod
    # def run_experiment(
    #     self, dataset, models, dataset_name: str, num_data_points: int
    # ) -> None:
    #     """
    #     This method should implement the running of the experiment.

    #     :param dataset: The dataset to run the experiment on.
    #     :param models: The models to use for the experiment.
    #     :param num_data_points: The number of data points to run the experiment on. If not specified uses all datapoints.
    #     :param dataset_name: The name of the dataset, must be a string that is directly accessible
    #     :return: None
    #     """
    #     pass


class MLflowLoggingCallback(Callback):
    """Pytorch Callback to ensure that system metrics are tracked and config data logged."""

    def __init__(self, config, experiment_name: str):
        super().__init__()
        self.config = config
        self.experiment_name = experiment_name

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(self.experiment_name)

            # attach to current mlflow run
            mlflow.start_run(run_id=trainer.logger.run_id)
            mlflow.log_text(OmegaConf.to_yaml(self.config, resolve=True), "config.yaml")
            logger.info(
                f"mlflow fluent context run for run id {trainer.logger.run_id} now active."
            )

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            mlflow.end_run()
            logger.info("mlflow fluent context sucessfull shutdown.")
