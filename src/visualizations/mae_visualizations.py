import matplotlib.pyplot as plt
import torch
import numpy as np

from pytorch_lightning import LightningModule


def _apply_inverse_fourier_transform(input_image: torch.Tensor) -> torch.Tensor:
    recon_shifted = torch.fft.ifftshift(input_image, dim=(-2, -1))
    real_space_reconstruction = torch.fft.ifft2(
        recon_shifted, dim=(-2, -1), norm="ortho"
    )
    return torch.fft.fftshift(real_space_reconstruction, dim=(-2, -1))


def visualize_mae_results(model: LightningModule, batch_x: torch.Tensor, num_images=3):
    """
    Plot original input, masked input and reconstructed version of input.

    :param model: trained masked autoencoder model
    :param batch_x: Batch of images of type [B, 1, H, W]
    :param num_images: Optional input to specify the number of displayed instances.
    Must not be greater than batch size
    """
    num_images = min(num_images, batch_x.shape[0])
    model.eval()

    with torch.no_grad():
        batch_x = batch_x.to(model.device)
        preds, _, mask_img, _ = model(batch_x)

        # retransform prediction to 2d image
        pred_img = model.unpatchify(preds)

        # if necessary expand mask to patch size
        if pred_img.size() != mask_img.size():
            mask_expanded = mask_img.unsqueeze(-1).repeat(1, 1, model.pixels_per_patch)
            mask_img = model.unpatchify(mask_expanded)

        # build final reconstruction and masked version
        reconstruction = pred_img * mask_img + batch_x * (1 - mask_img)
        masked_input = batch_x * (1 - mask_img)

        # calculate real space versions
        real_space_reconstruction = _apply_inverse_fourier_transform(reconstruction)
        original_real_space_reconstruction = _apply_inverse_fourier_transform(batch_x)

    fig, axes = plt.subplots(num_images, 8, figsize=(25, 4 * num_images))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        orig_img = batch_x[i][0].cpu().numpy()
        mask_img_plot = masked_input[i][0].cpu().numpy()
        recon_img = reconstruction[i][0].cpu().numpy()

        realspace_magnitude = real_space_reconstruction[i][0].abs().cpu().numpy()
        orig_realspace_magnitude = (
            original_real_space_reconstruction[i][0].abs().cpu().numpy()
        )
        diff_magnitude = np.abs(orig_realspace_magnitude - realspace_magnitude)
        real_part = real_space_reconstruction[i][0].real.cpu().numpy()
        imag_part = real_space_reconstruction[i][0].imag.cpu().numpy()

        # original hologram
        axes[i][0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
        axes[i][0].set_title("Original")
        axes[i][0].axis("off")

        # masked hologram
        axes[i][1].imshow(mask_img_plot, cmap="gray", vmin=0, vmax=1)
        axes[i][1].set_title(f"Masked ({(model.hparams.mask_ratio*100):.0f}%)")
        axes[i][1].axis("off")

        # reconstructed hologram fourier
        axes[i][2].imshow(recon_img, cmap="gray", vmin=0, vmax=1)
        axes[i][2].set_title("Fourier Space")
        axes[i][2].axis("off")

        # original real space magnitude
        axes[i][3].imshow(orig_realspace_magnitude, cmap="gray", vmin=0, vmax=1)
        axes[i][3].set_title("Original Real Space Magnitude")
        axes[i][3].axis("off")

        # real space magnitude
        axes[i][4].imshow(realspace_magnitude, cmap="gray", vmin=0, vmax=1)
        axes[i][4].set_title("Real Space Magnitude")
        axes[i][4].axis("off")

        # show difference image between original real space magnitude and reconstructed real space magnitude
        vmax_diff = np.percentile(diff_magnitude, 99.0)
        vmax_diff = vmax_diff if vmax_diff > 0 else 1e-5
        axes[i][5].imshow(diff_magnitude, cmap="hot", vmin=0, vmax=vmax_diff)
        axes[i][5].set_title("Difference Real Space Magnitude")
        axes[i][5].axis("off")

        # real space - Real part
        vmax_real = np.percentile(np.abs(real_part), 99.0)
        axes[i][6].imshow(real_part, cmap="RdBu", vmin=-vmax_real, vmax=vmax_real)
        axes[i][6].set_title("Real Space (Real Part)")
        axes[i][6].axis("off")

        # real space - Imaginary part
        vmax_imag = np.percentile(np.abs(imag_part), 99.0)
        axes[i][7].imshow(imag_part, cmap="RdBu", vmin=-vmax_imag, vmax=vmax_imag)
        axes[i][7].set_title("Real Space (Imaginary Part)")
        axes[i][7].axis("off")

    plt.tight_layout()
    return fig
