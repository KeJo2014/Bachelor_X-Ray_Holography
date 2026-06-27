import torch
import torch.nn as nn


class CenterFocusedTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        sigma: float = 0.5,
        smooth: float = 1e-6,
    ):
        """
        alpha: False positive weight
        beta: False negative weight
        sigma: parameters that adjust the importance decay from center to edges
        """
        super().__init__()  # TODO: Rationalize Parameter Selection
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.smooth = smooth

    def _get_spatial_weights(
        self, shape: torch.Size, device: torch.device
    ) -> torch.Tensor:
        """Creates 2d gaussian curve for weighting center to edges"""
        _, _, H, W = shape

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        weights = torch.exp(-(x_grid**2 + y_grid**2) / (2 * self.sigma**2))
        return weights.unsqueeze(0).unsqueeze(0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        weights = self._get_spatial_weights(logits.shape, logits.device)
        dim = (1, 2, 3)

        true_positives = (probs * targets * weights).sum(dim=dim)
        false_positives = (probs * (1 - targets) * weights).sum(dim=dim)
        false_negatives = ((1 - probs) * targets * weights).sum(dim=dim)

        # calc Tversky score
        tversky_score = (true_positives + self.smooth) / (
            true_positives
            + self.alpha * false_positives
            + self.beta * false_negatives
            + self.smooth
        )
        return (1.0 - tversky_score).mean()
