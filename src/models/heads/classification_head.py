import torch.nn as nn


class ViTMultiLabelClassificationHead(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, num_classes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),  # TODO: Hyperparam
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        :param x: input features with shape [B, Num_Patches, Embed_Dim]
        """
        # calculate global average pooling over all patches
        x_pooled = x.mean(dim=1)
        logits = self.mlp(x_pooled)
        return logits
