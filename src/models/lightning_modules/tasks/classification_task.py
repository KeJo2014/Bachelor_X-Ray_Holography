import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from metrics.multi_label_classification_metrics import get_metric_collection
from torchmetrics.classification import MultilabelConfusionMatrix


class LitClassificationTask(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
        num_classes: int,
        freeze_encoder: bool = False,
        lr: float = 1e-4,
        encoder_lr: float = 1e-5,
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

        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.test_conf_mat = MultilabelConfusionMatrix(
            num_labels=self.hparams.num_classes
        )

    def train(self, mode: bool = True):
        """
        overwrites lighting train method to ensure that backbone can stay frozen
        """
        super().train(mode)
        if self.hparams.freeze_encoder:
            self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """extracts features"""
        if self.hparams.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        logits = self.head(features)
        return logits

    def _shared_eval_step(self, batch, batch_idx, metrics_collection, prefix: str):
        x, y, _ = batch
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        metrics_collection.update(logits, y.long())

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
        x, y, _ = batch
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.train_metrics.update(logits, y.long())

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
        x, y, _ = batch
        logits = self(x)
        self.test_conf_mat.update(logits, y.long())
        return self._shared_eval_step(batch, batch_idx, self.test_metrics, "test")

    def configure_optimizers(self):
        if self.hparams.freeze_encoder:
            # optimize only the head
            optimizer_groups = [
                {"params": self.head.parameters(), "lr": self.hparams.lr}
            ]
        else:
            # full-finetuning
            optimizer_groups = [
                {"params": self.encoder.parameters(), "lr": self.hparams.encoder_lr},
                {"params": self.head.parameters(), "lr": self.hparams.lr},
            ]

        optimizer = torch.optim.AdamW(
            optimizer_groups,
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
