import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

from pytorch_lightning import LightningModule


def _apply_inverse_fourier_transform(input_image: torch.Tensor) -> torch.Tensor:
    """Helper function to apply inverse Fourier transform to a batch of images."""
    recon_shifted = torch.fft.ifftshift(input_image, dim=(-2, -1))
    real_space_reconstruction = torch.fft.ifft2(
        recon_shifted, dim=(-2, -1), norm="ortho"
    )
    return torch.fft.fftshift(real_space_reconstruction, dim=(-2, -1))


def visualize_backbone_results(
    model: LightningModule, batch_x: torch.Tensor, num_images=3
):
    """
    Plot original input, masked input and reconstructed version of input.
    Supports 1-channel and 3-channel (CL, CR, Diff) inputs.

    :param model: trained masked autoencoder model
    :param batch_x: Batch of images of type [B, C, H, W]
    :param num_images: Optional input to specify the number of displayed instances.
    Must not be greater than batch size
    """
    num_images = min(num_images, batch_x.shape[0])
    C = batch_x.shape[1]
    model.eval()

    with torch.no_grad():
        batch_x = batch_x.to(model.device)
        preds, _, mask_img, _ = model(batch_x)
        pred_img = model.unpatchify(preds)

        # Build final reconstruction and masked version
        reconstruction = pred_img * mask_img + batch_x * (1 - mask_img)
        masked_input = batch_x * (1 - mask_img)

        # Calculate real space versions
        real_space_reconstruction = _apply_inverse_fourier_transform(reconstruction)
        original_real_space_reconstruction = _apply_inverse_fourier_transform(batch_x)

    cols = 8 if C == 1 else 10
    fig, axes = plt.subplots(num_images, cols, figsize=(3.5 * cols, 4 * num_images))
    if num_images == 1:
        axes = [axes]

    def _plot_ax(ax, img, title, cmap="gray", vmin=0, vmax=1):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    mask_ratio_pct = int(getattr(model.hparams, "mask_ratio", 0) * 100)

    for i in range(num_images):
        if C == 1:
            orig_img = batch_x[i][0].cpu().numpy()
            mask_img_plot = masked_input[i][0].cpu().numpy()
            recon_img = reconstruction[i][0].cpu().numpy()

            realspace_mag = real_space_reconstruction[i][0].abs().cpu().numpy()
            orig_realspace_mag = (
                original_real_space_reconstruction[i][0].abs().cpu().numpy()
            )
            diff_mag = np.abs(orig_realspace_mag - realspace_mag)

            real_part = real_space_reconstruction[i][0].real.cpu().numpy()
            imag_part = real_space_reconstruction[i][0].imag.cpu().numpy()

            vmax_diff = max(np.percentile(diff_mag, 99.0), 1e-5)
            vmax_real = np.percentile(np.abs(real_part), 99.0)
            vmax_imag = np.percentile(np.abs(imag_part), 99.0)

            _plot_ax(axes[i][0], orig_img, "Original", vmin=0, vmax=1)
            _plot_ax(
                axes[i][1], mask_img_plot, f"Masked ({mask_ratio_pct}%)", vmin=0, vmax=1
            )
            _plot_ax(axes[i][2], recon_img, "Fourier Space")
            _plot_ax(axes[i][3], orig_realspace_mag, "Original Real Space Magnitude")
            _plot_ax(axes[i][4], realspace_mag, "Real Space Magnitude")
            _plot_ax(
                axes[i][5],
                diff_mag,
                "Difference Real Space Magnitude",
                cmap="hot",
                vmax=vmax_diff,
            )
            _plot_ax(
                axes[i][6],
                real_part,
                "Real Space (Real Part)",
                cmap="RdBu",
                vmin=-vmax_real,
                vmax=vmax_real,
            )
            _plot_ax(
                axes[i][7],
                imag_part,
                "Real Space (Imaginary Part)",
                cmap="RdBu",
                vmin=-vmax_imag,
                vmax=vmax_imag,
            )

        else:
            orig_cl = batch_x[i][0].cpu().numpy()
            recon_cl = reconstruction[i][0].cpu().numpy()

            orig_cr = batch_x[i][1].cpu().numpy()
            recon_cr = reconstruction[i][1].cpu().numpy()

            orig_diff = batch_x[i][2].cpu().numpy()
            masked_diff = masked_input[i][2].cpu().numpy()
            recon_diff = reconstruction[i][2].cpu().numpy()

            realspace_mag = real_space_reconstruction[i][2].abs().cpu().numpy()
            orig_realspace_mag = (
                original_real_space_reconstruction[i][2].abs().cpu().numpy()
            )
            diff_mag = np.abs(orig_realspace_mag - realspace_mag)

            max_orig_diff = max(abs(orig_diff.min()), abs(orig_diff.max()))
            vmax_error = max(np.percentile(diff_mag, 99.0), 1e-5)

            # CL & CR structure
            _plot_ax(axes[i][0], orig_cl, "Orig CL", cmap="viridis", vmin=0, vmax=1)
            _plot_ax(axes[i][1], recon_cl, "Recon CL", cmap="viridis", vmin=0, vmax=1)

            _plot_ax(axes[i][2], orig_cr, "Orig CR", cmap="viridis", vmin=0, vmax=1)
            _plot_ax(axes[i][3], recon_cr, "Recon CR", cmap="viridis", vmin=0, vmax=1)

            # diff holo
            _plot_ax(
                axes[i][4],
                orig_diff,
                "Orig Diff",
                cmap="coolwarm",
                vmin=-max_orig_diff,
                vmax=max_orig_diff,
            )
            _plot_ax(
                axes[i][5],
                masked_diff,
                f"Masked Diff ({mask_ratio_pct}%)",
                cmap="coolwarm",
                vmin=-max_orig_diff,
                vmax=max_orig_diff,
            )
            _plot_ax(
                axes[i][6],
                recon_diff,
                "Recon Diff",
                cmap="coolwarm",
                vmin=-max_orig_diff,
                vmax=max_orig_diff,
            )

            # real space error based on diff channel
            _plot_ax(
                axes[i][7], orig_realspace_mag, "Orig Real (Diff)", vmin=None, vmax=None
            )
            _plot_ax(
                axes[i][8], realspace_mag, "Recon Real (Diff)", vmin=None, vmax=None
            )
            _plot_ax(
                axes[i][9], diff_mag, "Real Error (Diff)", cmap="hot", vmax=vmax_error
            )

    plt.tight_layout()
    return fig


def plot_multiclass_confusion_matrix(matrix: torch.Tensor, class_names: list = None):
    """
    Plot confusion matrix for a multi-class classification problem.
    """
    matrix_np = matrix.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        matrix_np,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    plt.tight_layout()
    return fig
