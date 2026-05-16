import torch
import torch.nn.functional as F

from models.lightning_modules.pretext_tasks.pretext_task_action import PretextTaskAction


class RandomMaskingStrategy(PretextTaskAction):
    def __init__(
        self, img_size: int = 960, patch_size: int = 16, mask_ratio: float = 0.75
    ):
        super().__init__(
            img_size=img_size, patch_size=patch_size, mask_ratio=mask_ratio
        )

    def generate_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """Creates a random boolean mask."""
        rand_tensor = torch.rand(batch_size, num_patches, device=device)
        return rand_tensor < self.mask_ratio

    def compute_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, mask_1d: torch.Tensor
    ) -> torch.Tensor:
        """Calculates MSE loss only on masked patches."""
        return F.mse_loss(preds[mask_1d], targets[mask_1d])
