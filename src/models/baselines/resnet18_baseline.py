import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import timm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from metrics.multi_label_classification_metrics import get_metric_collection
from torchmetrics.classification import MultilabelConfusionMatrix


class LitResnetBaseline(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        pretrained: bool = False,
        channels: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.C = channels
        self.test_conf_mat = MultilabelConfusionMatrix(
            num_labels=self.hparams.num_classes
        )

        self.model = timm.create_model(
            "resnet18",
            pretrained=self.hparams.pretrained,
            num_classes=self.hparams.num_classes,
            in_chans=channels,
        )

        metrics = get_metric_collection(num_classes=self.hparams.num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_eval_step(self, batch, metrics_collection, prefix: str):
        x, y, _ = batch
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        metrics_collection.update(torch.sigmoid(logits), y.long())

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

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        self.test_conf_mat.update(logits, y.long())
        return self._shared_eval_step(batch, self.test_metrics, "test")

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
