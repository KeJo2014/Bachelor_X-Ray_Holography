import torch
import torch.nn.functional as F

from torch.amp import autocast
from models.lightning_modules.pretext_tasks.pretext_task_action import PretextTaskAction


class RandomMaskingRealSpaceStrategy(PretextTaskAction):
    def __init__(
        self, img_size: int = 960, patch_size: int = 16, mask_ratio: float = 0.75
    ):
        super().__init__(
            img_size=img_size, patch_size=patch_size, mask_ratio=mask_ratio
        )
        self.grid_size = img_size // patch_size

    def generate_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """Generates a random boolean mask."""
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
        """Calculate real-space MSE loss over the complete reconstruction hologram."""

        with autocast(device_type="cuda", enabled=False):
            preds_f32 = preds.float()
            targets_f32 = targets.float()

            reconstructed_patches = targets_f32.clone()
            reconstructed_patches[mask_1d] = preds_f32[mask_1d]
            reconstructed_img = self._unpatchify(reconstructed_patches)
            target_img = self._unpatchify(targets_f32)

            # shift lower frequency parts from center to the edges
            recon_shifted = torch.fft.ifftshift(reconstructed_img, dim=(-2, -1))
            target_shifted = torch.fft.ifftshift(target_img, dim=(-2, -1))

            # apply IFFFT
            recon_real_space = torch.fft.ifft2(
                recon_shifted, dim=(-2, -1), norm="ortho"
            )
            target_real_space = torch.fft.ifft2(
                target_shifted, dim=(-2, -1), norm="ortho"
            )

            # reverse frequency part shift
            recon_real_space = torch.fft.fftshift(recon_real_space, dim=(-2, -1))
            target_real_space = torch.fft.fftshift(target_real_space, dim=(-2, -1))

            recon_mag = torch.abs(recon_real_space)
            target_mag = torch.abs(target_real_space)

            loss = F.mse_loss(recon_mag, target_mag)

        return loss
