import torch

from models.loss_functions import RadiallyWeightedLoss
from models.lightning_modules.pretext_tasks.pretext_task_action import PretextTaskAction


class RandomMaskingStrategy(PretextTaskAction):
    def __init__(
        self,
        img_size: int = 960,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        centrosymmetric: bool = True,
        custom_loss: bool = True,
    ):
        """
        Handles the masking strategy, including mask generation, unpatchifying, and loss computation.
        """

        super().__init__(
            img_size=img_size, patch_size=patch_size, mask_ratio=mask_ratio
        )
        self.loss_function = RadiallyWeightedLoss(loss_type="l2")
        self.grid_size = img_size // patch_size
        self.centrosymmetric = centrosymmetric
        self.custom_loss = custom_loss

    def generate_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        num_mask = int(num_patches * self.mask_ratio)

        if not self.centrosymmetric:
            noise = torch.rand(batch_size, num_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            mask = torch.zeros(batch_size, num_patches, device=device, dtype=torch.bool)
            for i in range(batch_size):
                mask[i, ids_shuffle[i, :num_mask]] = True
            return mask

        else:
            half_patches = num_patches // 2
            half_mask_count = num_mask // 2

            noise = torch.rand(batch_size, half_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)

            rand_half = torch.zeros(
                batch_size, half_patches, device=device, dtype=torch.bool
            )
            for i in range(batch_size):
                rand_half[i, ids_shuffle[i, :half_mask_count]] = True

            mirrored_half = torch.flip(rand_half, dims=[1])
            if num_patches % 2 != 0:
                center_mask = torch.full(
                    (batch_size, 1),
                    (num_mask % 2 != 0),
                    device=device,
                    dtype=torch.bool,
                )
                return torch.cat([rand_half, center_mask, mirrored_half], dim=1)
            else:
                return torch.cat([rand_half, mirrored_half], dim=1)

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
        """Calculates Loss. For RGB mode: Decouples structure (MSE) and magnetism (Radial)."""

        B, N = mask_1d.shape
        H_patches = self.img_size // self.patch_size
        W_patches = self.img_size // self.patch_size

        mask_2d_grid = mask_1d.view(B, 1, H_patches, W_patches).float()
        mask_2d_full = mask_2d_grid.repeat_interleave(self.patch_size, dim=2)
        loss_mask = mask_2d_full.repeat_interleave(self.patch_size, dim=3)

        pred_img = self._unpatchify(preds)
        target_img = self._unpatchify(targets)

        C = pred_img.shape[1]

        if C == 3:
            mse_loss_fn = torch.nn.MSELoss(reduction="none")

            loss_cl = (
                mse_loss_fn(pred_img[:, 0:1], target_img[:, 0:1]) * loss_mask
            ).sum() / (loss_mask.sum() + 1e-8)
            loss_cr = (
                mse_loss_fn(pred_img[:, 1:2], target_img[:, 1:2]) * loss_mask
            ).sum() / (loss_mask.sum() + 1e-8)

            if getattr(self, "custom_loss", True):
                loss_diff = self.loss_function(
                    pred_img[:, 2:3], target_img[:, 2:3], mask=loss_mask
                )
                total_loss = loss_cl + loss_cr + (5.0 * loss_diff)
            else:
                loss_diff = (
                    mse_loss_fn(pred_img[:, 2:3], target_img[:, 2:3]) * loss_mask
                ).sum() / (loss_mask.sum() + 1e-8)
                total_loss = loss_cl + loss_cr + loss_diff
            return total_loss
        else:
            return self.loss_function(pred_img, target_img, mask=loss_mask)
