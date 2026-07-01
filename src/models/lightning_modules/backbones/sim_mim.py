import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from typing import Tuple
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from models.lightning_modules.pretext_tasks.pretext_task_action import PretextTaskAction


class LitSimMIM(pl.LightningModule):
    def __init__(
        self,
        pretext_strategy: PretextTaskAction,
        img_size: int = 960,
        patch_size: int = 16,
        embed_dim: int = 768,
        mask_ratio: float = 0.75,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.1,
        channels: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.C = channels
        self.pretext_strategy = pretext_strategy
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        self.pixels_per_patch = patch_size * patch_size * self.C
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.pixels_per_patch))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.encoder = timm.create_model(
            "vit_base_patch8_224",
            img_size=img_size,
            patch_size=patch_size,
            pretrained=False,
            num_classes=0,
            global_pool="",
            in_chans=self.C,
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.pixels_per_patch),
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Disects 2d image [B, C, H, W] into 1d patches [B, Num_Patches, Patch_Size^2]."""
        p = self.patch_size
        h = w = self.grid_size
        x = imgs.reshape(shape=(imgs.shape[0], self.C, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, (p**2) * self.C))
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Transforms a sequence of patches [B, Num_Patches, Patch_Size^2] back to an image [B, C, H, W]."""
        p = self.patch_size
        h = w = self.grid_size

        x = patches.reshape(shape=(patches.shape[0], h, w, p, p, self.C))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(patches.shape[0], self.C, h * p, w * p))
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        patches = self.patchify(x)

        mask_1d = self.pretext_strategy.generate_mask(B, self.num_patches, x.device)

        # replace masked patches with learnable tokens
        mask_tokens = self.mask_token.expand(B, self.num_patches, -1)
        mask_bool = mask_1d.unsqueeze(-1)
        patches_masked = torch.where(mask_bool, mask_tokens, patches)

        x_masked_img = self.unpatchify(patches_masked)
        mask_img = F.interpolate(
            mask_1d.view(B, 1, self.grid_size, self.grid_size).float(),
            size=(self.hparams.img_size, self.hparams.img_size),
            mode="nearest",
        )

        features = self.encoder(x_masked_img)
        patch_tokens = features[:, 1:, :]
        preds = self.decoder(patch_tokens)

        return preds, mask_1d, mask_img, x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _, _ = batch
        preds, mask_1d, _, _ = self(x)
        targets = self.patchify(x)

        loss = self.pretext_strategy.compute_loss(preds, targets, mask_1d)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _, _ = batch
        preds, mask_1d, _, _ = self(x)
        targets = self.patchify(x)

        loss = self.pretext_strategy.compute_loss(preds, targets, mask_1d)

        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        preds, mask_1d, _, _ = self(x)
        targets = self.patchify(x)

        loss = self.pretext_strategy.compute_loss(preds, targets, mask_1d)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

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
