import matplotlib.pyplot as plt

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