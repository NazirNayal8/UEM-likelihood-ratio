import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from scipy.stats import wasserstein_distance

def process_input(x, y, device):
    x = x.to(device)

    if x.ndim == 3:
        x = x.unsqueeze(0)
    y = torch.tensor(y).cuda().unsqueeze(0)

    return x, y


@torch.no_grad()
def compute_log_likelihood(prior, x, y, feature_extractor):
    # write a description for this function
    """
    Args:   
        prior: PixelCNN prior
        x: input image
        y: label

    Returns:
        likelihood: log likelihood of the input image
    """

    x, y = process_input(x, y, prior.device)
    x = x.unsqueeze(0)
    if feature_extractor is not None:
        x = feature_extractor(x)
    B = x.shape[0]
    vq_vae = prior.vq_vae

    vq_outputs = vq_vae.vq(
        vq_vae.pre_vq_conv(vq_vae.encoder(x))
    )
    encoding_indices = vq_outputs.encoding_indices[:, 0] # (BN)
    encoding_indices = encoding_indices.view(B, -1).contiguous().long() # (B, N)

    # get the prior
    logits = prior(encoding_indices)  

    log_probs = logits.log_softmax(dim=-1)
    B, N, C = log_probs.shape

    log_probs = log_probs.reshape(B * N, C)

    likelihood = log_probs.gather(
        dim=1, index=encoding_indices.view(-1).unsqueeze(1)).sum(dim=0)

    return -likelihood


@torch.no_grad()
def compute_log_likelihood_v2(prior, x, y):
    # write a description for this function
    """
    Args:   
        prior: PixelCNN prior
        x: input image
        y: label

    Returns:
        likelihood: log likelihood of the input image
    """

    x, y = process_input(x, y, prior.device)

    vq_vae = prior.vq_vae
    codebook_outputs = vq_vae.vq(
        vq_vae.pre_vq_conv(vq_vae.encoder(x)))

    B, C, H, W = codebook_outputs.quantized.shape

    encoding_indices = codebook_outputs.encoding_indices[:, 0].long() 
    encoding_indices = encoding_indices.view(B, H, W)

    # get the prior
    logits = prior(encoding_indices, label=y)  # shape: B, C, H, W
    logits = logits.permute(0, 2, 3, 1).contiguous()  # shape: B, H, W, C

    likelihood = F.cross_entropy(
        logits.view(-1, prior.hparams.MODEL.INPUT_DIM), encoding_indices.view(-1))

    return likelihood


def compute_dataset_log_likelihood(prior, dataset, max_samples=10000, arbitrary_cls=None, feature_extractor=None):

    num_samples = min(len(dataset), max_samples)
    likelihoods = np.zeros(num_samples)
    for i in tqdm(range(num_samples)):

        x, y = dataset[i]

        # this condition is used for when we want to compute the likelihood of an
        # OoD sample but with choosing an explicit class to condition on.
        if arbitrary_cls is not None:
            assert isinstance(
                arbitrary_cls, int), "arbitrary_cls must be an integer"
            y = arbitrary_cls
        
        likelihoods[i] = compute_log_likelihood(prior, x, y, feature_extractor=feature_extractor).cpu().item()

    return likelihoods


def compute_dataset_log_likelihood_batched(prior, dataset, batch_size=1, num_workers=4, max_samples=10000, arbitrary_cls=None):


    num_samples = min(len(dataset), max_samples)
    dataloader = DataLoader(Subset(dataset, np.arange(num_samples)), batch_size=batch_size,
                            shuffle=False, num_workers=4)
    likelihoods = []
    for i, (x, y) in enumerate(tqdm(dataloader)):
 
        # this condition is used for when we want to compute the likelihood of an
        # OoD sample but with choosing an explicit class to condition on.
        if arbitrary_cls is not None:
            assert isinstance(
                arbitrary_cls, int), "arbitrary_cls must be an integer"
            y = arbitrary_cls
        likelihoods.extend([compute_log_likelihood(prior, x, y).view(-1).cpu().numpy()])
    likelihoods = np.concatenate(likelihoods)
    return likelihoods


@torch.no_grad()
def get_perplexity(prior, x, y):
    """
    Args:
        prior: PixelCNN prior
        x: input image
        y: label

    Returns:
        perplexity: perplexity of the input image
    """
    x, y = process_input(x, y, prior.device)

    vq_vae = prior.vq_vae
    codebook_outputs = vq_vae(x)

    return codebook_outputs.perplexity    


@torch.no_grad()
def get_dataset_perplexity(prior, dataset, max_samples=10000, arbitrary_cls=None):
    """
    Args:
        prior: prior model, assumes has a vq_vae attribute
        dataset: dataset to compute the perplexity on
        max_samples: maximum number of samples to compute the perplexity on
        arbitrary_cls: if not None, compute the perplexity of the dataset by
            conditioning on this class.

    Returns:
        perplexities: perplexity of the dataset
    """
    num_samples = min(len(dataset), max_samples)
    perplexities = np.zeros(num_samples)
    for i in tqdm(range(num_samples)):

        x, y = dataset[i]

        if arbitrary_cls is not None:
            assert isinstance(
                arbitrary_cls, int), "arbitrary_cls must be an integer"
            y = arbitrary_cls

        perplexities[i] = get_perplexity(prior, x, y).cpu().item()

    return perplexities


@torch.no_grad()
def extract_distances_to_probs(prior, dataset, probs, num_tokens, distance_func="kl", max_size=10000):
     # write a description for this function
    """
    Args:
        prior: PixelCNN prior
        dataset: dataset to compute the perplexity on
        probs: probabilities of the tokens
        num_tokens: number of tokens
        distance_func: distance function to use
        max_size: maximum number of samples to compute the perplexity on

    Returns:
        distances: distances of the dataset
    """
    dataset_len = min(len(dataset), max_size)

    distances = np.zeros(dataset_len)

    for i in tqdm(range(dataset_len)):

        x, y = dataset[i]

        z = prior.vq_vae.pre_vq_conv(prior.vq_vae.encoder(x.cuda().unsqueeze(0)))
        outputs = prior.vq_vae.vq(z)
        indices = outputs.encoding_indices.flatten().cpu().numpy()
        
        counts = np.bincount(indices, minlength=num_tokens) 
        p = counts / counts.sum()

        if distance_func == "kl":
            distances[i] = np.sum(p * np.log((p / probs) + 10-8))
        elif distance_func == "l1":
            distances[i] = np.sum(np.abs(p - probs))
        elif distance_func == "l2":
            distances[i] = np.sum((p - probs)**2)
        elif distance_func == "emd":
            distances[i] = wasserstein_distance(p, probs)


    return distances

@torch.no_grad()
def compute_codebook_usage_distribution(prior, dataset, codebook_size):
    """
    Args:
        prior: a prior model, assumes has a vq_vae attribute
        dataset: dataset to compute the perplexity on
        codebook_size: size of the codebook

    Returns:
        codebook_usage: distribution of the codebook
    """
    dataset_len = len(dataset)
    codebook_usage = np.zeros(codebook_size)

    for i in tqdm(range(dataset_len)):

        x, y = dataset[i]

        z = prior.vq_vae.pre_vq_conv(prior.vq_vae.encoder(x.cuda().unsqueeze(0)))
        outputs = prior.vq_vae.vq(z)
        indices = outputs.encoding_indices.flatten().cpu().numpy()
        
        counts = np.bincount(indices, minlength=codebook_size) 
        codebook_usage += counts

    codebook_usage = codebook_usage / codebook_usage.sum()

    return codebook_usage


@torch.no_grad()
def compute_codebook_usage_distribution_2gram(prior, dataset, codebook_size, order_type, max_size=10000):
    """
    Args:
        prior: a prior model, assumes has a vq_vae attribute
        dataset: dataset to compute the perplexity on
        codebook_size: size of the codebook
        ordered: whether to count the 2-grams in order or not
        max_size: maximum number of samples to compute the perplexity on

    Returns:
        codebook_usage: distribution of the codebook
    """
    dataset_len = min(len(dataset), max_size)

    counts = np.zeros((codebook_size, codebook_size))

    for i in tqdm(range(dataset_len)):

        x, y = dataset[i]

        z = prior.vq_vae.pre_vq_conv(prior.vq_vae.encoder(x.cuda().unsqueeze(0)))
        outputs = prior.vq_vae.vq(z)
        indices = outputs.encoding_indices.flatten().cpu().numpy()

        for j in range(len(indices) - 1):
            
            if order_type == "bag_of_words":
                for k in range(j + 1, len(indices)):
                    a = max(indices[j], indices[k])
                    b = min(indices[j], indices[k])
                    counts[a, b] += 1
            elif order_type == "ordered":
                counts[indices[j], indices[j+1]] += 1
            elif order_type == "unordered":
                a = max(indices[j], indices[j+1])
                b = min(indices[j], indices[j+1])
                counts[a, b] += 1

    if order_type in ["unordered", "bag_of_words"]:
        counts = counts[np.tril_indices(codebook_size, k=0)]
    elif order_type == "ordered":
        counts = counts.flatten()
    
    counts = counts / counts.sum()
    return counts

@torch.no_grad()
def extract_distances_to_probs_2gram(prior, dataset, probs, codebook_size, order_type, distance_func="kl", max_size=10000):
    """
    Args:
        prior: PixelCNN prior
        dataset: dataset to compute the perplexity on
        probs: probabilities of the tokens
        num_tokens: number of tokens
        distance_func: distance function to use
        max_size: maximum number of samples to compute the perplexity on

    Returns:
        distances: distances of the dataset
    """
    dataset_len = min(len(dataset), max_size)

    distances = np.zeros(dataset_len)

    for i in tqdm(range(dataset_len)):

        x, y = dataset[i]

        z = prior.vq_vae.pre_vq_conv(prior.vq_vae.encoder(x.cuda().unsqueeze(0)))
        outputs = prior.vq_vae.vq(z)
        indices = outputs.encoding_indices.flatten().cpu().numpy()

        count = np.zeros((codebook_size, codebook_size))
        for j in range(len(indices) - 1):
            if order_type == "bag_of_words":
                for k in range(j + 1, len(indices)):
                    a = max(indices[j], indices[k])
                    b = min(indices[j], indices[k])
                    count[a, b] += 1
            elif order_type == "ordered":
                count[indices[j], indices[j+1]] += 1
            elif order_type == "unordered":
                a = max(indices[j], indices[j+1])
                b = min(indices[j], indices[j+1])
                count[a, b] += 1

        if order_type in ["unordered", "bag_of_words"]:
            count = count[np.tril_indices(codebook_size, k=0)]
        elif order_type == "ordered":
            count = count.flatten()
        
        count = count / count.sum()

        if distance_func == "kl":
            distances[i] = np.sum(count * np.log((count / probs) + 10-8))
        elif distance_func == "l1":
            distances[i] = np.sum(np.abs(count - probs))
        elif distance_func == "l2":
            distances[i] = np.sum((count - probs)**2)
        elif distance_func == "emd":
            distances[i] = wasserstein_distance(count, probs)
        else:
            raise ValueError(f"Unknown distance function: {distance_func}")
    
    return distances

    
    