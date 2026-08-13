import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def create_2d_tsne_plot(reduced_embeddings_2d, str_labels):
    fig_static, ax_static = plt.subplots(figsize=(10, 8))
    unique_labels = list(set(str_labels))
    cmap = plt.get_cmap("tab10", len(unique_labels))

    for idx, cls_name in enumerate(unique_labels):
        cls_indices = [i for i, lbl in enumerate(str_labels) if lbl == cls_name]
        ax_static.scatter(
            reduced_embeddings_2d[cls_indices, 0],
            reduced_embeddings_2d[cls_indices, 1],
            color=cmap(idx),
            label=cls_name,
            alpha=0.7,
        )

    ax_static.legend(title="Classes")
    ax_static.set_title("t-SNE Projection of Extracted Features (2D)")
    return fig_static


def create_3d_tsne_visualization(reduced_embeddings_3d, str_labels):
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
