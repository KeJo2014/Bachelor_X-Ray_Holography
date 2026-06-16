import torch
import torch.nn as nn
import torch.nn.functional as F


class RadiallyWeightedLoss(nn.Module):
    def __init__(self, loss_type="l1"):
        """
        :param loss_type: 'l1' for Mean Absolute Error, 'l2' for Mean Squared Error
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, mask=None):
        B, C, H, W = pred.shape
        device = pred.device

        # calculate base loss
        if self.loss_type == "l1":
            base_loss = F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "l2":
            base_loss = F.mse_loss(pred, target, reduction="none")
        else:
            raise ValueError("loss_type should be either 'l1' or 'l2'")

        # create a radial weighting matrix
        center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0

        y_coords = torch.arange(H, device=device, dtype=torch.float32) - center_y
        x_coords = torch.arange(W, device=device, dtype=torch.float32) - center_x
        Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # calculate euclidian distance for each pixel
        R = torch.sqrt(X**2 + Y**2).unsqueeze(0).unsqueeze(0)

        # determine valid pixels (only those who were masked before)
        valid_pixels = torch.ones_like(R)
        if mask is not None:
            valid_pixels = mask.float()

        # noralize weights
        R_valid = R * valid_pixels
        mean_R = R_valid.sum() / (valid_pixels.sum() + 1e-8)
        R_normalized = R / mean_R

        weighted_loss = base_loss * R_normalized * valid_pixels
        final_loss = weighted_loss.sum() / (valid_pixels.sum() + 1e-8)

        return final_loss
