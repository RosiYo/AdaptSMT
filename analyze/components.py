"""
Functions to perform dimensionality reduction on embeddings and visualize them.
author: Adrián Roselló Pedraza (RosiYo)
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def extract_pca_image_from_embeddings(groups_embeddings: dict, filename: str):
    """
    Perform PCA on the embeddings and save an image of the first two principal components,
    coloring points based on their group.

    Args:
        groups_embeddings (dict): A dictionary where keys are group names (str) and values are torch.Tensor embeddings.
        filename (str): The filename to save the image to.
    """
    # Combine embeddings and keep track of group labels
    embeddings_list = []
    labels = []
    for group_name, embeddings in groups_embeddings.items():
        embeddings_np = embeddings.detach().cpu().numpy()
        embeddings_list.append(embeddings_np)
        labels.extend([group_name] * embeddings_np.shape[0])

    embeddings_combined = np.vstack(embeddings_list)
    labels = np.array(labels)

    # Perform PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    components = pca.fit_transform(embeddings_combined)

    # Plot the first two principal components with different colors for each group
    plt.figure(figsize=(8, 6))
    unique_groups = np.unique(labels)
    for group in unique_groups:
        idx = labels == group
        plt.scatter(components[idx, 0], components[idx, 1], label=group, alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def extract_tsne_image_from_embeddings(
    groups_embeddings: dict, filename: str, perplexity: float = 30.0
):
    """
    Perform T-SNE on the embeddings and save an image of the 2D projection,
    coloring points based on their group.

    Args:
        groups_embeddings (dict): A dictionary where keys are group names (str) and values are torch.Tensor embeddings.
        filename (str): The filename to save the image to.
        perplexity (float): The perplexity parameter for T-SNE.
    """
    # Combine embeddings and keep track of group labels
    embeddings_list = []
    labels = []
    for group_name, embeddings in groups_embeddings.items():
        embeddings_np = embeddings.detach().cpu().numpy()
        embeddings_list.append(embeddings_np)
        labels.extend([group_name] * embeddings_np.shape[0])

    embeddings_combined = np.vstack(embeddings_list)
    labels = np.array(labels)

    # Perform T-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    components = tsne.fit_transform(embeddings_combined)

    # Plot the T-SNE components with different colors for each group
    plt.figure(figsize=(8, 6))
    unique_groups = np.unique(labels)
    for group in unique_groups:
        idx = labels == group
        plt.scatter(components[idx, 0], components[idx, 1], label=group, alpha=0.7)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("T-SNE Projection")
    plt.legend()
    plt.savefig(filename)
    plt.close()
