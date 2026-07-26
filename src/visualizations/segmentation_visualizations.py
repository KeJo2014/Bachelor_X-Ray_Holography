import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter, shift
from scipy.signal.windows import hann
from matplotlib.widgets import Button


def visualize_segmentation_result(x, true_mask, predicted_mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(x, cmap="gray")
    axes[0].set_title("Original Hologramm")
    axes[0].axis("off")

    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("Model Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def reconstruct_and_evaluate_magnetic_pattern(
    diff_holo: np.ndarray,
    seg_mask: np.ndarray | None,
    roi: tuple = None,
    phase: float = 0.0,
    mask_sigma: float = 10.0,
    scale: tuple = (1.0, 99.0),
    title_prefix: str = "",
) -> np.ndarray:
    if seg_mask is not None:
        transmission_mask = 1.0 - seg_mask.astype(float)
        if mask_sigma > 0:
            transmission_mask = gaussian_filter(transmission_mask, sigma=mask_sigma)

        masked_holo = diff_holo * transmission_mask
    else:
        masked_holo = diff_holo

    window_y = hann(masked_holo.shape[0])
    window_x = hann(masked_holo.shape[1])
    window_2d = np.outer(window_y, window_x)
    masked_holo = masked_holo * window_2d

    reconstruction = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(masked_holo)))

    # apply phase
    reconstruction = reconstruction * np.exp(1j * phase)
    if roi is not None:
        ys, ye, xs, xe = roi
        reconstruction = reconstruction[ys:ye, xs:xe]

    real_part = np.real(reconstruction)
    imag_part = np.imag(reconstruction)
    abs_part = np.abs(reconstruction)

    mi_real, ma_real = np.percentile(real_part, scale)
    mi_imag, ma_imag = np.percentile(imag_part, scale)

    max_val_real = max(abs(mi_real), abs(ma_real))
    max_val_imag = max(abs(mi_imag), abs(ma_imag))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{title_prefix} (Sigma: {mask_sigma})", fontsize=14, fontweight="bold"
    )

    axs[0].imshow(abs_part, cmap="viridis")
    axs[0].set_title("Absolute value")
    axs[0].axis("off")

    im1 = axs[1].imshow(real_part, cmap="RdBu", vmin=-max_val_real, vmax=max_val_real)
    axs[1].set_title(f"real part (magnetic)\nPhase: {phase:.2f} rad")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(imag_part, cmap="RdBu", vmin=-max_val_imag, vmax=max_val_imag)
    axs[2].set_title("imaginary part")
    axs[2].axis("off")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show(block=False)

    return reconstruction


def magnetic_pattern_visualization(cl, cr, mask=None):
    if mask is not None:
        mask = 1.0 - mask
        symmetric_mask = mask * np.flip(mask)  # ensure centrosymmetricity
        soft_mask = gaussian_filter(symmetric_mask.astype(float), sigma=10)

        cl = cl * soft_mask
        cr = cr * soft_mask

    diff_holo = cl - cr
    real_space_complex = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(diff_holo)))

    real_space_mag = np.abs(real_space_complex)
    real_space_real = np.real(real_space_complex)

    plt.figure(figsize=(12, 5), dpi=150)

    # patterson map
    plt.subplot(1, 2, 1)
    plt.title("Patterson Map (Log Magnitude)")
    plt.imshow(np.log10(real_space_mag + 1e-12), cmap="inferno")
    plt.colorbar(fraction=0.046, pad=0.04)

    # real part -> magnetic pattern
    plt.subplot(1, 2, 2)
    plt.title("Magnetic Pattern (Real Part)")
    limit = np.percentile(np.abs(real_space_real), 99.9)
    plt.imshow(real_space_real, cmap="RdBu_r", vmin=-limit, vmax=limit)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import pytorch_lightning as pl
    from datasets.simulated_dataset import HologramDataModule

    pl.seed_everything(44)

    data_path = "C:/Users/kelle/Documents/storage/xray/Raw_holo_sim/reduced2"
    current_mode = "rgb"

    data_module = HologramDataModule(
        data_path,
        mode=current_mode,
        total_samples=5000,
        add_poisson_noise=False,
        num_workers=0,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    inv_label_map = {v: k for k, v in data_module.label_map.items()}

    batch_iterator = iter(train_loader)
    fig_control, ax_control = plt.subplots(figsize=(4, 1.5))
    fig_control.canvas.manager.set_window_title("Dataloader Steuerung")
    ax_control.axis("off")

    ax_btn = fig_control.add_axes([0.1, 0.2, 0.8, 0.6])
    btn_next = Button(ax_btn, "Next")

    def show_next(event):
        try:
            batch = next(batch_iterator)
            x_batch, holo_pattern_batch, mask_batch = batch

            x_img = x_batch[0].numpy()
            mask_img = mask_batch[0].squeeze().numpy()

            cl = x_img[0]
            cr = x_img[1]

            # simulate stronger imperfections
            intensity_factor = 1.05
            cr_imperfect = cr * intensity_factor

            # subpixel drift
            cr_imperfect = shift(cr_imperfect, shift=(1.0, -0.5), order=3)
            diff_holo_real_world = cl - cr_imperfect

            for f in plt.get_fignums():
                if plt.figure(f) != fig_control:
                    plt.close(f)

            reconstruct_and_evaluate_magnetic_pattern(
                diff_holo=diff_holo_real_world,
                seg_mask=None,
                phase=0.0,
                mask_sigma=0.0,
                scale=(1.0, 99.0),
                title_prefix="Without mask",
            )

            reconstruct_and_evaluate_magnetic_pattern(
                diff_holo=diff_holo_real_world,
                seg_mask=mask_img,
                phase=0.0,
                mask_sigma=10.0,
                scale=(1.0, 99.0),
                title_prefix="With mask",
            )

        except StopIteration:
            btn_next.label.set_text("Done")
            plt.draw()

    btn_next.on_clicked(show_next)
    show_next(None)
    plt.show()
