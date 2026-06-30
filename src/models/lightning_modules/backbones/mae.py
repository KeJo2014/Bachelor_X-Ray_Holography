import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

from typing import Tuple
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from models.lightning_modules.pretext_tasks.pretext_task_action import PretextTaskAction


class LitMAE(pl.LightningModule):
    def __init__(
        self,
        pretext_strategy: PretextTaskAction,
        img_size: int = 960,
        patch_size: int = 16,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mask_ratio: float = 0.75,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.1,
        channels: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pretext_strategy = pretext_strategy
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.pixels_per_patch = patch_size * patch_size * channels
        self.C = channels

        self.encoder = timm.create_model(
            "vit_base_patch8_224",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=self.C,
            pretrained=False,
            num_classes=0,
            global_pool="",
        )

        # DECODER
        # projection from encoder to decoder space
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )

        self.decoder_blocks = nn.ModuleList(
            [
                timm.models.vision_transformer.Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # final projection to pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, self.pixels_per_patch, bias=True
        )

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Cuts image of shape [B, C, H, W] in patches [B, Num_Patches, Patch_Size^2 * C]."""
        p = self.patch_size
        h = w = self.grid_size
        x = imgs.reshape(shape=(imgs.shape[0], self.C, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.C))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs image from patches.
        """
        p = self.patch_size
        h = w = self.grid_size

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.C))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.C, h * p, w * p))

        return imgs

    def forward_encoder(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]

        x = self.encoder.patch_embed(x)

        # add positional embeddings without class token
        x = x + self.encoder.pos_embed[:, 1:, :]

        # mask and drop masked patches
        noise = torch.rand(B, self.num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        len_keep = int(self.num_patches * (1 - self.hparams.mask_ratio))
        ids_keep = ids_shuffle[:, :len_keep]

        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )

        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_visible), dim=1)

        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        return x, ids_restore, ids_keep

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        B = x.shape[0]
        x = self.decoder_embed(x)

        cls_token = x[:, :1, :]
        x_visible = x[:, 1:, :]

        # generate masked tokens for the masked patches
        num_masked = self.num_patches - x_visible.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)

        # concat masked token in original order
        x_ = torch.cat([x_visible, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2])
        )

        # add CLS token again
        x = torch.cat([cls_token, x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x[:, 1:, :])
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, ids_restore, ids_keep = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)

        B = x.shape[0]
        mask_1d = torch.ones([B, self.num_patches], device=x.device)
        mask_1d[:, : ids_keep.shape[1]] = 0
        mask_1d = torch.gather(mask_1d, dim=1, index=ids_restore)

        mask_img = F.interpolate(
            mask_1d.view(B, 1, self.grid_size, self.grid_size).float(),
            size=(self.hparams.img_size, self.hparams.img_size),
            mode="nearest",
        )

        return pred, mask_1d, mask_img, x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, label, dataset_mask = batch
        pred, mask_1d, _, _ = self(x)
        target = self.patchify(x)

        loss = self.pretext_strategy.compute_loss(pred, target, mask_1d.bool())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, label, dataset_mask = batch
        pred, mask_1d, _, _ = self(x)
        target = self.patchify(x)

        loss = self.pretext_strategy.compute_loss(pred, target, mask_1d.bool())

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, label, dataset_mask = batch
        pred, mask_1d, _, _ = self(x)
        target = self.patchify(x)

        loss = self.pretext_strategy.compute_loss(pred, target, mask_1d.bool())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
