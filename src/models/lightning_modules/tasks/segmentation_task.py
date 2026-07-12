import pytorch_lightning as pl
import torch
from metrics.segmentation_metrics import get_metric_collection
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from models.loss_functions import CenterFocusedTverskyLoss


class LitSegmentationTask(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
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
        self.loss_function = (
            CenterFocusedTverskyLoss()
        )  # TODO: Adjust Parameters -> Hyperparams?

        # can freeze backbone
        if self.hparams.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # setup label calculation
        metrics = get_metric_collection()

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

    def _shared_eval_step(self, batch, batch_idx, metrics_collection, prefix: str):
        x, _, y_mask = batch
        logits = self(x)

        loss = self.loss_function(logits, y_mask)
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

        loss = self.loss_function(logits, y_mask)
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
        head_params = set(self.head.parameters())
        
        if self.hparams.freeze_encoder:
            optimizer_groups = [
                {"params": list(head_params), "lr": self.hparams.lr}
            ]
        else:
            encoder_params_unique = [
                p for p in self.encoder.parameters() if p not in head_params
            ]
            
            optimizer_groups = [
                {"params": encoder_params_unique, "lr": self.hparams.encoder_lr},
                {"params": list(head_params), "lr": self.hparams.lr},
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
