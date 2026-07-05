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
    ):
        """
        Handles the masking strategy, including mask generation, unpatchifying, and loss computation.

        :param img_size: Spatial dimensions of the input image in pixels.
        :param patch_size: Spatial dimensions of a single image patch.
        :param mask_ratio: Fraction of patches to be masked during training.
        :param centrosymmetric: If True, applies a centrosymmetric mask to prevent trivial shortcuts.
        """

        super().__init__(
            img_size=img_size, patch_size=patch_size, mask_ratio=mask_ratio
        )
        self.loss_function = RadiallyWeightedLoss(loss_type="l2")
        self.grid_size = img_size // patch_size
        self.centrosymmetric = centrosymmetric

    def generate_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """Creates a boolean mask, either purely random or centrosymmetric."""

        if not self.centrosymmetric:
            rand_tensor = torch.rand(batch_size, num_patches, device=device)
            return rand_tensor < self.mask_ratio

        else:
            half_patches = num_patches // 2
            rand_half = (
                torch.rand(batch_size, half_patches, device=device) < self.mask_ratio
            )
            mirrored_half = torch.flip(rand_half, dims=[1])
            if num_patches % 2 != 0:
                center_patch = (
                    torch.rand(batch_size, 1, device=device) < self.mask_ratio
                )
                return torch.cat([rand_half, center_patch, mirrored_half], dim=1)
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

            loss_diff = self.loss_function(
                pred_img[:, 2:3], target_img[:, 2:3], mask=loss_mask
            )

            total_loss = loss_cl + loss_cr + (5.0 * loss_diff)
            return total_loss

        else:
            return self.loss_function(pred_img, target_img, mask=loss_mask)
