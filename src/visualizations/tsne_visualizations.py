import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from matplotlib.lines import Line2D


def create_2d_tsne_plot(reduced_embeddings_2d, str_labels):
    """Create a static 2D t-SNE plot using Matplotlib."""

    fig_static, ax_static = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(str_labels))
    cmap = plt.get_cmap("Set1")

    for idx, cls_name in enumerate(unique_labels):
        cls_indices = [i for i, lbl in enumerate(str_labels) if lbl == cls_name]
        ax_static.scatter(
            reduced_embeddings_2d[cls_indices, 0],
            reduced_embeddings_2d[cls_indices, 1],
            color=cmap(idx),
            label=cls_name,
            alpha=0.7,
        )

    ax_static.tick_params(axis="both", which="major", labelsize=20)
    # ax_static.legend(title="Classes")
    # ax_static.set_title("t-SNE Projection of Extracted Features (2D)")
    plt.tight_layout()
    return fig_static


def create_2d_contour_plot(reduced_embeddings_2d, str_labels):
    """Create a 2D density contour plot for t-SNE embeddings."""

    fig_contour, ax_contour = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(str_labels))
    cmap = plt.get_cmap("Set1")
    legend_handles = []

    for idx, cls_name in enumerate(unique_labels):
        cls_indices = [i for i, lbl in enumerate(str_labels) if lbl == cls_name]
        x_data = reduced_embeddings_2d[cls_indices, 0]
        y_data = reduced_embeddings_2d[cls_indices, 1]

        sns.kdeplot(
            x=x_data,
            y=y_data,
            ax=ax_contour,
            color=cmap(idx),
            alpha=0.8,
            levels=6,
            linewidths=2.5,
        )
        custom_line = Line2D([0], [0], color=cmap(idx), lw=2.5, label=cls_name)
        legend_handles.append(custom_line)

    ax_contour.tick_params(axis="both", which="major", labelsize=20)
    # ax_contour.legend(handles=legend_handles, title="Classes")
    # ax_contour.set_title("2D Density Contour Plot of Extracted Features")
    plt.tight_layout()
    return fig_contour


def create_orthogonal_projections(reduced_embeddings_3d, str_labels):
    """Create 3 orthogonal 2D projections from 3D t-SNE embeddings."""
    fig_ortho, axes = plt.subplots(1, 3, figsize=(18, 5))
    unique_labels = sorted(set(str_labels))
    cmap = plt.get_cmap("Set1")

    projections = [
        (0, 1, "t-SNE 1 (X)", "t-SNE 2 (Y)"),
        (1, 2, "t-SNE 2 (Y)", "t-SNE 3 (Z)"),
        (0, 2, "t-SNE 1 (X)", "t-SNE 3 (Z)"),
    ]

    for ax, (dim1, dim2, xlabel, ylabel) in zip(axes, projections):
        for idx, cls_name in enumerate(unique_labels):
            cls_indices = [i for i, lbl in enumerate(str_labels) if lbl == cls_name]

            ax.scatter(
                reduced_embeddings_3d[cls_indices, dim1],
                reduced_embeddings_3d[cls_indices, dim2],
                color=cmap(idx),
                label=cls_name,
                alpha=0.3,
                s=3,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Projection: {xlabel} vs. {ylabel}")
        # ax.legend(title="Classes", loc="lower right", markerscale=2.5)

    plt.tight_layout()
    return fig_ortho


def create_3d_tsne_visualization(reduced_embeddings_3d, str_labels):
    """Create an interactive 3D t-SNE plot using Plotly."""
    df_tsne = pd.DataFrame(
        {
            "tsne_1": reduced_embeddings_3d[:, 0],
            "tsne_2": reduced_embeddings_3d[:, 1],
            "tsne_3": reduced_embeddings_3d[:, 2],
            "label": str_labels,
        }
    )

    fig_interactive = px.scatter_3d(
        df_tsne,
        x="tsne_1",
        y="tsne_2",
        z="tsne_3",
        color="label",
        title="t-SNE Projection of Extracted Features",
        labels={"label": "Classification Label"},
        hover_data=["label"],
    )
    fig_interactive.update_traces(marker=dict(size=4))
    return fig_interactive
