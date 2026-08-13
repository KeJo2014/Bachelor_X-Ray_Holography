import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from metrics.segmentation_metrics import get_metric_collection
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class LitUnetBaseline(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
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

    def _shared_eval_step(self, batch, batch_idx, metrics_collection, prefix: str):
        x, _, y_mask = batch
        logits = self(x)

        loss = self.loss_fn(logits, y_mask)
        probs = torch.sigmoid(logits)

        metrics_collection.update(probs, (y_mask > 0.5).long())

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(
            metrics_collection,
            on_step=False,
            on_epoch=True,
            prog_bar=(prefix == "val"),
        )
        return loss

    def training_step(self, batch, batch_idx):
        x, _, y_mask = batch
        logits = self(x)

        loss = self.loss_fn(logits, y_mask)
        probs = torch.sigmoid(logits)

        self.train_metrics.update(probs, (y_mask > 0.5).long())

        self.log(
            f"train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, self.test_metrics, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        warmup = LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
