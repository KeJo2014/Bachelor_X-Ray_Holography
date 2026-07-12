import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter


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
    seg_mask: np.ndarray,
    roi: tuple = None,
    phase: float = 0.0,
    mask_sigma: float = 10.0,
    scale: tuple = (1.0, 99.0),
) -> np.ndarray:
    """
    Rekonstruiert das magnetische Muster aus einem differentiellen Hologramm
    unter Verwendung einer prädizierten Segmentierungsmaske.

    Args:
        diff_holo: Das differentielle Hologramm (Positiv - Negativ Helizität).
        seg_mask: Binäre Maske des Beamstops (1 = Beamstop/Ausblenden, 0 = Hologramm erhalten).
                  Achtung: Falls dein Modell 1 für Hintergrund vorhersagt, muss dies invertiert werden.
        roi: Region of Interest (y_start, y_end, x_start, x_end) für die magnetische Domäne.
        phase: Globale Phasenverschiebung (in Radiant) zur Trennung von Real-/Imaginärteil.
        mask_sigma: Sigma für den Gauss-Filter, um die Maskenkanten weich zu machen (verhindert Gibbs-Ringing).
        scale: Perzentile (min, max) für die Kontrast-Skalierung der Visualisierung.

    Returns:
        reconstruction: Das komplexwertige, rekonstruierte Realraum-Bild.
    """

    # 1. Maske vorbereiten (Weiche Kanten sind kritisch!)
    # Wir gehen davon aus, dass seg_mask == 1 bedeutet "Hier ist der Beamstop".
    # Daher wollen wir eine Transmissionsmaske, die überall 1 ist, außer am Beamstop (dort 0).
    transmission_mask = 1.0 - seg_mask.astype(float)

    if mask_sigma > 0:
        transmission_mask = gaussian_filter(transmission_mask, sigma=mask_sigma)

    # 2. Hologramm maskieren
    masked_holo = diff_holo * transmission_mask

    # 3. Fourier-Transformations-Rekonstruktion (FTH)
    # Beachte das korrekte Shifting vor und nach der iFFT
    reconstruction = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(masked_holo)))

    # 4. Phasenschieber anwenden (Rotiert Signal in den Real/Imaginärteil)
    reconstruction = reconstruction * np.exp(1j * phase)

    # 5. ROI zuschneiden (Optional, aber empfohlen, da das Bild meist größtenteils leer ist)
    if roi is not None:
        ys, ye, xs, xe = roi
        reconstruction = reconstruction[ys:ye, xs:xe]

    # 6. Visualisierung vorbereiten (Wissen aus fomocid adaptiert)
    real_part = np.real(reconstruction)
    imag_part = np.imag(reconstruction)
    abs_part = np.abs(reconstruction)

    # Symmetrische Skalierung für magnetischen Kontrast
    mi_real, ma_real = np.percentile(real_part, scale)
    mi_imag, ma_imag = np.percentile(imag_part, scale)

    # Maximale Amplitude für symmetrische Farbskala (divergierend) sichern
    max_val_real = max(abs(mi_real), abs(ma_real))
    max_val_imag = max(abs(mi_imag), abs(ma_imag))

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(abs_part, cmap="viridis")
    axs[0].set_title("Absolutbetrag (Struktur/Ladung)")
    axs[0].axis("off")

    # Magnetischer Kontrast liegt (je nach globaler Phase) im Realteil
    im1 = axs[1].imshow(real_part, cmap="RdBu", vmin=-max_val_real, vmax=max_val_real)
    axs[1].set_title(f"Realteil (Magnetisch)\nPhase: {phase:.2f} rad")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(imag_part, cmap="RdBu", vmin=-max_val_imag, vmax=max_val_imag)
    axs[2].set_title("Imaginärteil")
    axs[2].axis("off")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return reconstruction


def magnetic_pattern_visualization(cl, cr, mask=None):
    if mask is not None:
        mask = 1.0 - mask
        # 1. Weiche Kanten erzeugen, um das Kreuz-Ringing der Aufhängung zu verhindern.
        # mask sollte hier als float (0.0 bis 1.0) vorliegen.
        symmetric_mask = mask * np.flip(mask)  # ensure centrosymmetric
        soft_mask = gaussian_filter(symmetric_mask.astype(float), sigma=10)

        # 2. Multiplikativ anwenden (Zentrosymmetrie der Ausgangsmaske vorausgesetzt!)
        cl = cl * soft_mask
        cr = cr * soft_mask

    # Differenzhologramm bilden
    diff_holo = cl - cr

    # FFT in den Realraum
    real_space_complex = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(diff_holo)))

    real_space_mag = np.abs(real_space_complex)
    real_space_real = np.real(real_space_complex)

    plt.figure(figsize=(12, 5), dpi=150)

    # Plot 1: Patterson Map
    plt.subplot(1, 2, 1)
    plt.title("Patterson Map (Log Magnitude)")
    plt.imshow(np.log10(real_space_mag + 1e-12), cmap="inferno")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Plot 2: Magnetisches Muster (Realteil) mit Min-Max-Normalisierung
    # Die Min-Max-Skalierung auf das 99.9. Perzentil verhindert Kontrastverluste
    # und Verzerrungen, die durch eine einfache Standardisierung entstehen würden.
    plt.subplot(1, 2, 2)
    plt.title("Magnetic Pattern (Real Part)")
    limit = np.percentile(np.abs(real_space_real), 99.9)
    plt.imshow(real_space_real, cmap="RdBu_r", vmin=-limit, vmax=limit)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def verify_inputs(cl, cr, mask):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    # 1. CL Hologramm (Logarithmisch, um die schwachen Streusignale zu sehen)
    im0 = axes[0].imshow(np.log10(np.abs(cl) + 1e-12), cmap="viridis")
    axes[0].set_title("CL Hologram (Log)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. CR Hologramm
    im1 = axes[1].imshow(np.log10(np.abs(cr) + 1e-12), cmap="viridis")
    axes[1].set_title("CR Hologram (Log)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Die Maske
    im2 = axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Applied Mask")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Differenz (CL - CR) linear mit Min-Max-Skalierung
    diff = cl - cr
    limit = np.percentile(np.abs(diff), 99.9)
    im3 = axes[3].imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[3].set_title("Difference (CL - CR)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


from datasets.simulated_dataset import HologramDataModule
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from matplotlib.widgets import Button

if __name__ == "__main__":
    pl.seed_everything(44)

    data_path = "C:/Users/kelle/Documents/storage/xray/Raw_holo_sim"
    current_mode = "rgb"

    data_module = HologramDataModule(
        data_path,
        mode=current_mode,
        center_holograms=True,
        add_poisson_noise=False,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    inv_label_map = {v: k for k, v in data_module.train_dataset.label_map.items()}

    # 1. Iterator für den Dataloader erstellen
    batch_iterator = iter(train_loader)

    # --- Setup für das Steuerungsfenster ---
    fig_control, ax_control = plt.subplots(figsize=(4, 1.5))
    fig_control.canvas.manager.set_window_title("Dataloader Steuerung")
    ax_control.axis("off")

    ax_btn = fig_control.add_axes([0.1, 0.2, 0.8, 0.6])
    btn_next = Button(ax_btn, "Nächstes Element ➔")

    def show_next(event):
        try:
            # 2. Nächsten Batch ziehen
            batch = next(batch_iterator)
            x_batch, holo_pattern_batch, mask_batch = batch

            # 3. Erstes Bild extrahieren
            x_img = x_batch[0].numpy()
            mask_img = mask_batch[0].squeeze().numpy()

            # Kanäle extrahieren
            cl = x_img[0]
            cr = x_img[1]

            # 4. Differenzhologramm bilden (Positiv - Negativ)
            diff_holo = cl - cr

            # 5. Alte Plot-Fenster schließen, außer dem Steuerungsfenster
            for f in plt.get_fignums():
                if plt.figure(f) != fig_control:
                    plt.close(f)

            print("Lade nächstes Element und rekonstruiere...")

            # 6. Unsere Funktion für die Rekonstruktion aufrufen
            reconstruction = reconstruct_and_evaluate_magnetic_pattern(
                diff_holo=diff_holo,
                seg_mask=mask_img,
                phase=np.pi,  # Kann später zur Optimierung angepasst werden
                mask_sigma=10.0,  # Wichtig gegen Ringing Artefakte
                scale=(1.0, 99.0),
                # roi=(22, 65, 83, 125),
            )

            # Neue Plots zeichnen, ohne den Haupt-Thread zu blockieren
            plt.show(block=False)

        except StopIteration:
            print("Ende des Dataloaders erreicht!")
            btn_next.label.set_text("Fertig (Ende)")
            plt.draw()

    # Den Button-Klick verknüpfen
    btn_next.on_clicked(show_next)

    # Das erste Element direkt beim Start laden
    show_next(None)

    # Hauptschleife am Laufen halten
    plt.show()
