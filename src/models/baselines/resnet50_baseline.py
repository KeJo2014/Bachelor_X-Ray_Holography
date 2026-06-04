import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import timm
from metrics.multi_label_classification_metrics import get_metric_collection


class LitResnetBaseline(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        pretrained: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(
            "resnet18",
            pretrained=self.hparams.pretrained,
            num_classes=self.hparams.num_classes,
            in_chans=1,
        )

        metrics = get_metric_collection(num_classes=self.hparams.num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, batch_idx, metrics_collection, prefix: str):
        x, y, _ = batch
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        metrics_collection.update(logits, y.long())

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
