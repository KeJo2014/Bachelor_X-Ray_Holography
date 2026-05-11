import matplotlib.pyplot as plt
import torch

from pytorch_lightning import LightningModule


def visualize_mae_results(model: LightningModule, batch_x: torch.Tensor, num_images=3):
    """
    Plot original input, masked input and reconstructed version of input.

    :param model: trained masked autoencoder model
    :param batch_x: Batch of images of type [B, 1, H, W]
    :param num_images: Optional input to specify the number of displayed instances.
    Must not be greater than batch size
    """
    model.eval()

    with torch.no_grad():
        batch_x = batch_x.to(model.device)
        preds, mask_1d, mask_img, _ = model(batch_x)

        # retransform prediction to 2d image
        pred_img = model.unpatchify(preds)

        # build final reconstruction and masked version
        reconstruction = pred_img * mask_img + batch_x * (1 - mask_img)
        masked_input = batch_x * (1 - mask_img)

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    if num_images == 1:
        axes = [axes]

    for i in range(min(num_images, batch_x.shape[0])):
        orig_img = batch_x[i][0].cpu().numpy()
        mask_img_plot = masked_input[i][0].cpu().numpy()
        recon_img = reconstruction[i][0].cpu().numpy()

        # original hologram
        axes[i][0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
        axes[i][0].set_title("Original")
        axes[i][0].axis("off")

        # masked hologram
        axes[i][1].imshow(mask_img_plot, cmap="gray", vmin=0, vmax=1)
        axes[i][1].set_title(f"Masked Input ({(model.hparams.mask_ratio*100):.0f}%)")
        axes[i][1].axis("off")

        # reconstructed hologram
        axes[i][2].imshow(recon_img, cmap="gray", vmin=0, vmax=1)
        axes[i][2].set_title("Reconstruction")
        axes[i][2].axis("off")

    plt.tight_layout()
    return fig
