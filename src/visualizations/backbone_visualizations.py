import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _apply_inverse_fourier_transform(input_image: torch.Tensor) -> torch.Tensor:
    """Helper function to apply inverse Fourier transform to a batch of images."""
    recon_shifted = torch.fft.ifftshift(input_image, dim=(-2, -1))
    real_space_reconstruction = torch.fft.ifft2(
        recon_shifted, dim=(-2, -1), norm="ortho"
    )
    return torch.fft.fftshift(real_space_reconstruction, dim=(-2, -1))


def _compute_model_outputs(model, batch_x):
    """conduct forward pass for reconstruction."""
    model.eval()
    with torch.no_grad():
        batch_x = batch_x.to(model.device)
        preds, _, mask_img, _ = model(batch_x)
        pred_img = model.unpatchify(preds)
        reconstruction = pred_img * mask_img + batch_x * (1 - mask_img)
        masked_input = batch_x * (1 - mask_img)

        # Calculate real space versions
        real_space_recon = _apply_inverse_fourier_transform(reconstruction)
        orig_real_space_recon = _apply_inverse_fourier_transform(batch_x)

    mask_ratio_pct = int(getattr(model.hparams, "mask_ratio", 0) * 100)
    return (
        reconstruction,
        masked_input,
        real_space_recon,
        orig_real_space_recon,
        mask_ratio_pct,
    )


def _plot_ax(ax, img, title, cmap="gray", vmin=0, vmax=1, show_colorbar=False):
    """Helper function to plot an image on a given axis."""
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return im


def _add_zoom_box(
    ax, axins, x0, x1, y0, y1, connect_corners, edgecolor="black", linewidth=0.8
):
    """Add a zoom box and connecting lines between the main axis and the inset axis."""
    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=edgecolor, linewidth=linewidth
    )
    ax.add_patch(rect)
    box_corner = {
        "top-left": (x0, y0),
        "top-right": (x1, y0),
        "bottom-left": (x0, y1),
        "bottom-right": (x1, y1),
    }
    inset_corner = {
        "top-left": (0, 1),
        "top-right": (1, 1),
        "bottom-left": (0, 0),
        "bottom-right": (1, 0),
    }

    for corner_in, corner_box in connect_corners:
        con = ConnectionPatch(
            xyA=inset_corner[corner_in],
            coordsA=axins.transAxes,
            xyB=box_corner[corner_box],
            coordsB=ax.transData,
            edgecolor=edgecolor,
            linewidth=linewidth,
            axesA=axins,
            axesB=ax,
        )
        axins.add_artist(con)
    return rect


def _add_zoom_insets(ax, img_data, cmap="coolwarm"):
    H, W = img_data.shape
    hw = int(min(H, W) * 0.05)

    cy, cx = H // 2 + 15, W // 2 - 15
    oy, ox = int(H * 0.23), int(W * 0.28)

    # inset near center
    center_crop = img_data[cy - hw : cy + hw, cx - hw : cx + hw]
    c_max = max(abs(center_crop.min()), abs(center_crop.max()))

    axins_c = ax.inset_axes([0.02, 0.02, 0.35, 0.35])
    axins_c.imshow(img_data, cmap=cmap, vmin=-c_max, vmax=c_max)
    axins_c.set_xlim(cx - hw, cx + hw)
    axins_c.set_ylim(cy + hw, cy - hw)
    axins_c.set_xticks([])
    axins_c.set_yticks([])

    _add_zoom_box(
        ax,
        axins_c,
        cx - hw,
        cx + hw,
        cy - hw,
        cy + hw,
        connect_corners=[("top-left", "top-left"), ("bottom-right", "bottom-right")],
    )

    # inset on peripheral region
    outer_crop = img_data[oy - hw : oy + hw, ox - hw : ox + hw]
    o_max = max(abs(outer_crop.min()), abs(outer_crop.max()))

    axins_o = ax.inset_axes([0.63, 0.63, 0.35, 0.35])
    axins_o.imshow(img_data, cmap=cmap, vmin=-o_max, vmax=o_max)
    axins_o.set_xlim(ox - hw, ox + hw)
    axins_o.set_ylim(oy + hw, oy - hw)
    axins_o.set_xticks([])
    axins_o.set_yticks([])

    _add_zoom_box(
        ax,
        axins_o,
        ox - hw,
        ox + hw,
        oy - hw,
        oy + hw,
        connect_corners=[("top-left", "top-left"), ("bottom-left", "bottom-right")],
    )


def _plot_1_channel_row(
    axes_row, orig, masked, recon, real_recon, orig_real_recon, mask_ratio_pct
):
    """Plottet eine Zeile für ein 1-Kanal-Bild."""
    orig_img = orig[0].cpu().numpy()
    mask_img_plot = masked[0].cpu().numpy()
    recon_img = recon[0].cpu().numpy()

    realspace_mag = real_recon[0].abs().cpu().numpy()
    orig_realspace_mag = orig_real_recon[0].abs().cpu().numpy()
    diff_mag = np.abs(orig_realspace_mag - realspace_mag)

    real_part = real_recon[0].real.cpu().numpy()
    imag_part = real_recon[0].imag.cpu().numpy()

    vmax_diff = max(np.percentile(diff_mag, 99.0), 1e-5)
    vmax_real = np.percentile(np.abs(real_part), 99.0)
    vmax_imag = np.percentile(np.abs(imag_part), 99.0)

    _plot_ax(axes_row[0], orig_img, "Original", vmin=0, vmax=1)
    _plot_ax(axes_row[1], mask_img_plot, f"Masked ({mask_ratio_pct}%)", vmin=0, vmax=1)
    _plot_ax(axes_row[2], recon_img, "Fourier Space")
    _plot_ax(axes_row[3], orig_realspace_mag, "Original Real Space Magnitude")
    _plot_ax(axes_row[4], realspace_mag, "Real Space Magnitude")
    _plot_ax(
        axes_row[5],
        diff_mag,
        "Difference Real Space Magnitude",
        cmap="hot",
        vmax=vmax_diff,
        show_colorbar=True,
    )
    _plot_ax(
        axes_row[6],
        real_part,
        "Real Space (Real Part)",
        cmap="RdBu",
        vmin=-vmax_real,
        vmax=vmax_real,
    )
    _plot_ax(
        axes_row[7],
        imag_part,
        "Real Space (Imaginary Part)",
        cmap="RdBu",
        vmin=-vmax_imag,
        vmax=vmax_imag,
        show_colorbar=True,
    )


def _plot_3_channel_row(
    axes_row, orig, masked, recon, real_recon, orig_real_recon, mask_ratio_pct
):
    """Plots a row for a 3-channel image."""
    orig_cl = orig[0].cpu().numpy()
    recon_cl = recon[0].cpu().numpy()
    orig_cr = orig[1].cpu().numpy()
    recon_cr = recon[1].cpu().numpy()

    orig_diff = orig[2].cpu().numpy()
    masked_diff = masked[2].cpu().numpy()
    recon_diff = recon[2].cpu().numpy()

    realspace_mag = real_recon[2].abs().cpu().numpy()
    orig_realspace_mag = orig_real_recon[2].abs().cpu().numpy()
    diff_mag = np.abs(orig_realspace_mag - realspace_mag)

    max_orig_diff = max(abs(orig_diff.min()), abs(orig_diff.max()))
    vmax_error = max(np.percentile(diff_mag, 99.0), 1e-5)

    # CL & CR structure
    _plot_ax(
        axes_row[0],
        orig_cl,
        "Orig CL",
        cmap="viridis",
        vmin=0,
        vmax=1,
        show_colorbar=True,
    )
    _plot_ax(
        axes_row[1],
        recon_cl,
        "Recon CL",
        cmap="viridis",
        vmin=0,
        vmax=1,
        show_colorbar=True,
    )
    _plot_ax(
        axes_row[2],
        orig_cr,
        "Orig CR",
        cmap="viridis",
        vmin=0,
        vmax=1,
        show_colorbar=True,
    )
    _plot_ax(
        axes_row[3],
        recon_cr,
        "Recon CR",
        cmap="viridis",
        vmin=0,
        vmax=1,
        show_colorbar=True,
    )

    # Diff Holo (original) -> with Zoom Insets
    _plot_ax(
        axes_row[4],
        orig_diff,
        "Orig Diff",
        cmap="coolwarm",
        vmin=-max_orig_diff,
        vmax=max_orig_diff,
        show_colorbar=True,
    )
    _add_zoom_insets(axes_row[4], orig_diff, cmap="coolwarm")

    # Diff Holo (masked)
    _plot_ax(
        axes_row[5],
        masked_diff,
        f"Masked Diff ({mask_ratio_pct}%)",
        cmap="coolwarm",
        vmin=-max_orig_diff,
        vmax=max_orig_diff,
    )

    # Diff Holo (recon) -> with zoom insets
    _plot_ax(
        axes_row[6],
        recon_diff,
        "Recon Diff",
        cmap="coolwarm",
        vmin=-max_orig_diff,
        vmax=max_orig_diff,
        show_colorbar=True,
    )
    _add_zoom_insets(axes_row[6], recon_diff, cmap="coolwarm")

    # Real space error based on diff channel
    _plot_ax(axes_row[7], orig_realspace_mag, "Orig Real (Diff)", vmin=None, vmax=None)
    _plot_ax(axes_row[8], realspace_mag, "Recon Real (Diff)", vmin=None, vmax=None)
    _plot_ax(axes_row[9], diff_mag, "Real Error (Diff)", cmap="hot", vmax=vmax_error)


def visualize_backbone_results(model, batch_x: torch.Tensor, num_images=3):
    """Visualizes the results of the backbone model for a batch of images."""
    num_images = min(num_images, batch_x.shape[0])
    C = batch_x.shape[1]
    recon, masked_in, real_recon, orig_real_recon, mask_pct = _compute_model_outputs(
        model, batch_x
    )
    cols = 8 if C == 1 else 10
    fig, axes = plt.subplots(
        num_images, cols, figsize=(3.5 * cols, 4 * num_images), squeeze=False
    )

    for i in range(num_images):
        if C == 1:
            _plot_1_channel_row(
                axes[i],
                batch_x[i],
                masked_in[i],
                recon[i],
                real_recon[i],
                orig_real_recon[i],
                mask_pct,
            )
        else:
            _plot_3_channel_row(
                axes[i],
                batch_x[i],
                masked_in[i],
                recon[i],
                real_recon[i],
                orig_real_recon[i],
                mask_pct,
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
