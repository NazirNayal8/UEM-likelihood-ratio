import os
import numpy as np
import torch
import torch.nn.functional as F
from pycave.bayes import GaussianMixture as GaussianMixtureGPU
from sklearn.mixture import GaussianMixture as GaussianMixtureCPU
from .vis import unnormalize_image, plot_side_by_side
from tqdm import tqdm

def train_gmms(
        ind_dataset,
        ood_dataset,
        num_components_ind,
        num_components_ood,
        covariance_type_ind="diag",
        covariance_type_ood="diag",
        max_epochs=120,
        device="gpu",
        gpu_index=0,
):
    
    assert device in ["cpu", "gpu"], f"device must be one of [cpu, gpu], you gave me {device} though..."

    if device == "gpu" and not torch.cuda.is_available():
        raise ValueError("device is set to 'gpu', but no GPU is available")


    if device == "gpu":
        GaussianMixture = GaussianMixtureGPU
    
    devices = 1
    if gpu_index is not None:
        devices = [gpu_index]

    trainer_params = dict(
        accelerator="gpu",
        devices=devices,
        max_epochs=max_epochs
    )

    gmm_ind = GaussianMixture(
        num_components=num_components_ind, 
        covariance_type=covariance_type_ind, 
        trainer_params=trainer_params,
    )

    gmm_ood = GaussianMixture(
        num_components=num_components_ood, 
        covariance_type=covariance_type_ood, 
        trainer_params=trainer_params,
    )

    fitted_ind = gmm_ind.fit(ind_dataset)
    fitted_ood = gmm_ood.fit(ood_dataset)

    return fitted_ind, fitted_ood


def compute_likelihood_ratio(ind_model, ood_model, x):

    flat = True
    if len(x.shape) == 3:
        H, W, C = x.shape
        x = x.reshape(-1, C)
        flat = False

    ind_likelihood = -ind_model.score_samples(x)
    ood_likelihood = -ood_model.score_samples(x)

    ratio = ind_likelihood - ood_likelihood

    if not flat:
        ratio = ratio.reshape(H, W)

    return ratio


def plot_anomaly_maps_from_dinov2(gmm_ind, gmm_ood, dinov2, dataset, output_path, interpolation_mode="nearest", save=True):

    from .dino import get_features
    IMAGENET_MEAN, IMAGENET_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for i in range(len(dataset)):
        x, y = dataset[i]
        H_x, W_x = x.shape[1], x.shape[2]
        H_d, W_d = H_x // 14, W_x // 14

        features = get_features(dinov2, x)
        features = features.reshape(H_d, W_d, -1)
        
        features = F.interpolate(features.permute(2, 0, 1).unsqueeze(0), size=(H_x, W_x), mode=interpolation_mode).squeeze(0).permute(1, 2, 0)
        lr_ratio = compute_likelihood_ratio(gmm_ind, gmm_ood, features)
        x_norm = unnormalize_image(x, IMAGENET_MEAN, IMAGENET_STD)

        if save:        
            plot_side_by_side(x_norm, lr_ratio, path=f"{output_path}/img_{interpolation_mode}_{i}.png")   
        else:
            plot_side_by_side(x_norm, lr_ratio, path=None)

def evaluate_dataset_from_dinov2(gmm_ind, gmm_ood, dinov2, dataset, gpu_index=0):

    from .dino import get_features
    from analysis.ood import OODEvaluator

    anomaly_scores= []
    ood_gts = []

    for i in tqdm(range(len(dataset))):

        x, y = dataset[i]
        H_x, W_x = x.shape[1], x.shape[2]
        H_d, W_d = H_x // 14, W_x // 14

        features = get_features(dinov2, x, gpu_index=gpu_index)
        features = features.reshape(H_d, W_d, -1)
        # upsample features to original image size using nearest neighbor interpolation using F.interpolate
        features = F.interpolate(features.permute(2, 0, 1).unsqueeze(0), size=(H_x, W_x), mode="nearest").squeeze(0).permute(1, 2, 0)
        lr_ratio = compute_likelihood_ratio(gmm_ind, gmm_ood, features)

        anomaly_scores.append(lr_ratio)
        ood_gts.append(y)

        ood_evaluator = OODEvaluator(None, None, None)
        anomaly_scores = np.stack([x.reshape(-1) for x in anomaly_scores])
        ood_gts = np.stack([x.reshape(-1) for x in ood_gts])
        metrics = ood_evaluator.evaluate_ood(
            anomaly_score=anomaly_scores,
            ood_gts=ood_gts,
            verbose=False
        )

        return metrics


