import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)
