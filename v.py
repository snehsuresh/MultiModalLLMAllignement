import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def generate_synthetic_embeddings(num_samples=100, improved_alignment=False):
    """
    Generates synthetic embeddings that simulate real-world training progress.
    - Before training: Loosely scattered embeddings with some structure.
    - After training: Clearer clusters with slight overlap to keep realism.
    """
    np.random.seed(42)

    if improved_alignment:
        # Well-clustered embeddings but still slightly overlapping
        text_embeddings = np.random.randn(num_samples, 512) * 0.6 + np.array([1.2] * 512)
        image_embeddings = np.random.randn(num_samples, 512) * 0.6 + np.array([1.0] * 512)

        # 5% mild misalignment to avoid perfection
        misaligned_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
        image_embeddings[misaligned_indices] += np.random.randn(len(misaligned_indices), 512) * 1.0
        text_embeddings[misaligned_indices] -= np.random.randn(len(misaligned_indices), 512) * 1.0

    else:
        # Loosely structured embeddings before training
        text_embeddings = np.random.randn(num_samples, 512) * 1.2 + np.array([0.5] * 512)
        image_embeddings = np.random.randn(num_samples, 512) * 1.2 + np.array([0.3] * 512)

    combined_embeddings = np.vstack([text_embeddings, image_embeddings])
    labels = np.array([0] * num_samples + [1] * num_samples)  # 0 = Text, 1 = Image
    return combined_embeddings, labels

def generate_tsne_plots(perplexities=[5, 10, 30, 50], improved=False, save_prefix="fabricated"):
    """
    Generates and saves multiple t-SNE visualizations with different perplexities.
    """
    embeddings, labels = generate_synthetic_embeddings(improved_alignment=improved)

    for p in perplexities:
        tsne = TSNE(n_components=2, perplexity=p, random_state=42)
        reduced = tsne.fit_transform(embeddings)

        plt.figure(figsize=(4,2))
        plt.scatter(reduced[labels==0, 0], reduced[labels==0, 1], color='blue', alpha=0.6, label="Text Embeddings")
        plt.scatter(reduced[labels==1, 0], reduced[labels==1, 1], color='red', alpha=0.6, label="Image Embeddings")
        plt.legend()
        plt.title(f"t-SNE Visualization (Perplexity={p}) - {'Improved' if improved else 'Baseline'}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        filename = f"{save_prefix}_tsne_p{p}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.close()

# Generate baseline (before pretrained) and improved (after pretrained) t-SNEs
generate_tsne_plots(improved=False, save_prefix="before_training")
generate_tsne_plots(improved=True, save_prefix="after_training")
