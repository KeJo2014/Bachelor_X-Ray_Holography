import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from typing import Tuple
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from models.lightning_modules.pretext_tasks.pretext_task_action import PretextTaskAction


class Dinov3Backbone(pl.LightningModule):
    def __init__(
        self,
        pretext_strategy: PretextTaskAction,
        img_size: int = 960,
        patch_size: int = 16,
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
        embed_dim = 768
        self.pixels_per_patch = self.patch_size * self.patch_size * self.C
        self.grid_size = img_size // self.patch_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.pixels_per_patch))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.encoder = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
            in_chans=self.C,
            global_pool="",
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.pixels_per_patch),
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Disects 2d image [B, C, H, W] into 1d patches [B, Num_Patches, Patch_Size^2]."""
        b, c, h, w = imgs.shape
        p = self.patch_size
        h_grid = w_grid = self.grid_size

        x = imgs.reshape(shape=(b, c, h_grid, p, w_grid, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(b, h_grid * w_grid, p**2 * c))
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Transforms a sequence of patches [B, Num_Patches, Patch_Size^2] back to an image [B, C, H, W]."""
        h_grid = w_grid = self.grid_size
        b, _, d = patches.shape
        p = self.patch_size
        c = d // (p**2)

        x = patches.reshape(shape=(b, h_grid, w_grid, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(b, c, h_grid * p, w_grid * p))
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape

        h_grid, w_grid = h // self.patch_size, w // self.patch_size
        num_patches = h_grid * w_grid
        patches = self.patchify(x)

        mask_1d = self.pretext_strategy.generate_mask(b, num_patches, x.device)

        # replace masked patches with learnable tokens
        mask_tokens = self.mask_token.expand(b, num_patches, -1)
        mask_bool = mask_1d.unsqueeze(-1)
        patches_masked = torch.where(mask_bool, mask_tokens, patches)

        x_masked_img = self.unpatchify(patches_masked)

        mask_img = F.interpolate(
            mask_1d.view(b, 1, h_grid, w_grid).float(),
            size=(h, w),
            mode="nearest",
        )

        features = self.encoder(x_masked_img)

        patch_tokens = features[:, -num_patches:, :]
        preds = self.decoder(patch_tokens)
        return preds, mask_1d, mask_img, x

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, prefix: str
    ) -> torch.Tensor:
        x, _, _ = batch
        preds, mask_1d, _, cropped_x = self(x)
        targets = self.patchify(cropped_x)

        loss = self.pretext_strategy.compute_loss(preds, targets, mask_1d)

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

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
