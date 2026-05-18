import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DPTSegmentationHead(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        num_classes = 1
        self.needs_full_image = True

        self.smp_model = smp.DPT(
            encoder_name=encoder_name,  # needs to be fitting the base architecture
            encoder_weights=None,
            in_channels=1,
            classes=num_classes,
        )

    def inject_pretrained_encoder(self, timm_encoder: nn.Module):
        """
        Replace generic smp encoder with trained Backbone encoder.
        """
        self.smp_model.encoder.model = timm_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not next(self.smp_model.encoder.parameters()).requires_grad:
            with torch.no_grad():
                features = self.smp_model.encoder(x)
        else:
            features = self.smp_model.encoder(x)

        decoder_output = self.smp_model.decoder(*features)
        logits = self.smp_model.segmentation_head(decoder_output)
        return logits
