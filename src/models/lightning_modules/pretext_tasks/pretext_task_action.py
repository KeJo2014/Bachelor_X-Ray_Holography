import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class PretextTaskAction(nn.Module, ABC):
    """Base class for self-supervised pretext tasks."""

    def __init__(self, img_size: int, patch_size: int, mask_ratio: float) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    @abstractmethod
    def generate_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generates mask for data
        """
        pass

    @abstractmethod
    def compute_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, mask_1d: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss for the pretext task masking strategy
        """
        pass
