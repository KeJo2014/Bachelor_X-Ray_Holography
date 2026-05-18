import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class LitSegmentationTask(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
        freeze_encoder: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "head"])
        self.encoder = encoder
        self.head = head

        # can freeze backbone
        if self.hparams.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # setup label calculation
        metrics = MetricCollection(
            {"IoU": BinaryJaccardIndex(), "dice": BinaryF1Score()}
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def train(self, mode: bool = True):
        """
        overwrites lighting train method to ensure that backbone can stay frozen
        """
        super().train(mode)
        if self.hparams.freeze_encoder:
            self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """extracts features"""

        if hasattr(self.head, "needs_full_image") and self.head.needs_full_image:
            logits = self.head(x)
            return logits

        if self.hparams.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        num_patches = self.head.grid_size**2
        # remove all prefix token including CLS token
        patch_tokens = features[:, -num_patches:, :]

        logits = self.head(patch_tokens)
        return logits

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates Dice loss
        """

        # calculate dice loss
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum()
        dice_score = (2.0 * intersection + smooth) / (
            probs.sum() + targets.sum() + smooth
        )
        dice_loss = 1.0 - dice_score

        return dice_loss

    def _shared_step(self, batch, batch_idx, metrics_collection, prefix: str):
        x, _, y_mask = batch
        logits = self(x)

        loss = self.dice_loss(logits, y_mask)
        probs = torch.sigmoid(logits)
        metrics_collection.update(probs, y_mask)

        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
        # if frozen only optimze the head
        parameters_to_optimize = (
            self.head.parameters() if self.hparams.freeze_encoder else self.parameters()
        )

        optimizer = torch.optim.AdamW(
            parameters_to_optimize,
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
