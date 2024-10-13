import numpy as np
import torch
from tqdm import tqdm



def compute_codebook_usage_distribution(
    vq_vae,
    dataset,
    codebook_size,
    max_size=200000,
    device=None,
):
    """
    Args:
        vq_vae: a VQ VAE model (as implemented in this repo)
        dataset: a torch dataset
        codebook_size: size of the codebook
        max_size: maximum number of samples to use from the dataset
        device: device to use, if None, use cuda if available, otherwise cpu
    Returns:
        codebook_usage: distribution of the codebook index usage
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vq_vae.to(device)
    dataset_len = min(len(dataset), max_size)
    codebook_usage = np.zeros(codebook_size)

    for i in tqdm(range(dataset_len)):
        x, y = dataset[i]

        z = vq_vae.pre_vq_conv(vq_vae.encoder(x.to(device).unsqueeze(0)))
        outputs = vq_vae.vq(z)
        indices = outputs.encoding_indices.flatten().cpu().numpy()

        counts = np.bincount(indices, minlength=codebook_size)
        codebook_usage += counts

    codebook_usage = codebook_usage / codebook_usage.sum()

    return codebook_usage


def extract_distances_to_probs(
    vq_vae, 
    dataset, 
    probs, 
    codebook_size, 
    distance_func="kl", 
    max_size=10000,
    device=None,
):
    """
    Args:
        vq_vae: a VQ VAE model (as implemented in this repo)
        dataset: a torch dataset
        probs: probabilities of codebook usage
        codebook_size: number of tokens in the codebook
        distance_func: distance function to use
        max_size: maximum number of samples to compute the perplexity on
        device: device to use, if None, use cuda if available, otherwise cpu
    Returns:
        distances: distances of the dataset samples to given distribution
    """
    
    if distance_func == "emd":
        from scipy.stats import wasserstein_distance

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vq_vae.to(device)
    dataset_len = min(len(dataset), max_size)
    distances = np.zeros(dataset_len)

    for i in tqdm(range(dataset_len)):
        x, y = dataset[i]

        z = vq_vae.pre_vq_conv(vq_vae.encoder(x.to(device).unsqueeze(0)))
        outputs = vq_vae.vq(z)
        indices = outputs.encoding_indices.flatten().cpu().numpy()

        counts = np.bincount(indices, minlength=codebook_size)
        p = counts / counts.sum()

        if distance_func == "kl":
            distances[i] = np.sum(p * np.log((p / (probs + 1e-9)) + 10e-8))
        elif distance_func == "l1":
            distances[i] = np.sum(np.abs(p - probs))
        elif distance_func == "l2":
            distances[i] = np.sum((p - probs) ** 2)
        elif distance_func == "emd":
            distances[i] = wasserstein_distance(p, probs)

    return distances
