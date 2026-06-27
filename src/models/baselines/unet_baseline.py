import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from metrics.segmentation_metrics import get_metric_collection


class LitUnetBaseline(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        channels: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=self.hparams.encoder_name,
            encoder_weights=self.hparams.encoder_weights,
            in_channels=channels,
            classes=1,
            activation=None,
        )

        self.C = channels
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        metrics = get_metric_collection()
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, batch_idx, metrics_collection, prefix: str):
        x, _, y_mask = batch
        logits = self(x)

        loss = self.loss_fn(logits, y_mask)
        probs = torch.sigmoid(logits)

        metrics_collection.update(probs, y_mask.long())

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.train_metrics, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.test_metrics, "test")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_epoch=True, prog_bar=False)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
