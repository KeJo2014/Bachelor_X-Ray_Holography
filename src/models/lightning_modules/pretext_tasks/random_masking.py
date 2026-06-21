import torch
from models.loss_functions import RadiallyWeightedLoss
from models.lightning_modules.pretext_tasks.pretext_task_action import PretextTaskAction


class RandomMaskingStrategy(PretextTaskAction):
    def __init__(
        self, img_size: int = 960, patch_size: int = 16, mask_ratio: float = 0.75
    ):
        super().__init__(
            img_size=img_size, patch_size=patch_size, mask_ratio=mask_ratio
        )
        self.loss_function = RadiallyWeightedLoss(loss_type="l2")
        self.grid_size = img_size // patch_size

    def generate_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """Creates a random boolean mask."""
        rand_tensor = torch.rand(batch_size, num_patches, device=device)
        return rand_tensor < self.mask_ratio

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Retransform 1D patches to 2d image"""
        p = self.patch_size
        h = w = self.grid_size
        c = patches.shape[-1] // (p**2)

        x = patches.reshape(shape=(patches.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(patches.shape[0], c, h * p, w * p))
        return x

    def compute_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, mask_1d: torch.Tensor
    ) -> torch.Tensor:
        """Calculates Radially Weighted Loss only on masked patches."""

        B, N = mask_1d.shape
        H_patches = self.img_size // self.patch_size
        W_patches = self.img_size // self.patch_size

        mask_2d_grid = mask_1d.view(B, 1, H_patches, W_patches).float()
        mask_2d_full = mask_2d_grid.repeat_interleave(self.patch_size, dim=2)
        mask_2d_full = mask_2d_full.repeat_interleave(self.patch_size, dim=3)
        ignore_mask = 1.0 - mask_2d_full

        loss = self.loss_function(
            self._unpatchify(preds), self._unpatchify(targets), mask=ignore_mask
        )
        return loss
