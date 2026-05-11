import torch
import torch.nn as nn
import math


class ViTSegmentationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 1,
        img_size: int = 960,
        patch_size: int = 16,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        self.bottleneck = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # dynamic upsampling
        num_upsample_layers = int(math.log2(patch_size))

        layers = []
        in_channels = 256

        for _ in range(num_upsample_layers):
            out_channels = max(in_channels // 2, 16)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels, out_channels, kernel_size=2, stride=2
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        self.decoder = nn.Sequential(*layers)
        self.segmentation_head = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, padding=1
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = patch_tokens.shape

        x = patch_tokens.transpose(1, 2).contiguous()
        x = x.view(b, d, self.grid_size, self.grid_size)

        x = self.bottleneck(x)
        x = self.decoder(x)

        logits = self.segmentation_head(x)
        return logits
