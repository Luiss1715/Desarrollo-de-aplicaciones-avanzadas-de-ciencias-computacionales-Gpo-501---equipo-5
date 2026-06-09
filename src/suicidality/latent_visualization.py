from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .embeddings import DEFAULT_EMBEDDING_MODEL, TransformerEmbedder, prepare_texts
from .latent_pipeline import load_phase3_dataset
from .protocol_metrics import labels_from_frame


def plot_projection(
    projection: np.ndarray,
    labels: list[int],
    title: str,
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for value, name in [(0, "no"), (1, "yes")]:
        mask = np.asarray(labels) == value
        plt.scatter(projection[mask, 0], projection[mask, 1], label=name, alpha=0.7)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="is_suicide")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def generate_visualizations(
    csv_path: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    pca_output: str | Path = "reports/latent_space_pca.png",
    umap_output: str | Path = "reports/latent_space_umap.png",
    batch_size: int = 32,
    device: str | None = None,
) -> list[Path]:
    frame = load_phase3_dataset(csv_path)
    embeddings = TransformerEmbedder(model_name, device=device).encode(
        prepare_texts(frame), batch_size=batch_size
    )
    labels = labels_from_frame(frame)
    generated = []

    pca_path = Path(pca_output)
    plot_projection(PCA(n_components=2).fit_transform(embeddings), labels, "Latent space: PCA", pca_path)
    generated.append(pca_path)

    try:
        import umap
    except ImportError:
        return generated
    umap_path = Path(umap_output)
    projection = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    plot_projection(projection, labels, "Latent space: UMAP", umap_path)
    generated.append(umap_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Transformer embeddings to two dimensions")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--pca-out", default="reports/latent_space_pca.png")
    parser.add_argument("--umap-out", default="reports/latent_space_umap.png")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    args = parser.parse_args()
    generate_visualizations(
        args.csv,
        model_name=args.model_name,
        pca_output=args.pca_out,
        umap_output=args.umap_out,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
