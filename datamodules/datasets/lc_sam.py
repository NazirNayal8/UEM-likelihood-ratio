import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import read_label


class LCSAM(Dataset):
    def __init__(self, hparams, verbose=False):
        super().__init__()

        self.hparams = hparams

        folders = os.listdir(hparams.DATASET_ROOT)

        self.features = []
        self.labels = []

        ood_counts = []

        # if verbose, use tqdm for folder iteration
        if verbose:
            from tqdm import tqdm
            folders = tqdm(folders, desc="Loading SAM features")

        for f in folders:
            if f in hparams.EXCLUDED_FOLDERS:
                continue
            
            # if verbose, update tqdm desc with name of the folder
            if verbose:
                folders.set_description(f"Loading SAM features from {f}")

            splits = os.listdir(os.path.join(hparams.DATASET_ROOT, f))

            if hparams.MODE == "train":
                mode = "train"
                if "sam" in splits:
                    mode = ""

            elif hparams.MODE == "val":
                mode = "val"
                if mode not in splits:
                    continue

            elif hparams.MODE == "test":
                mode = "test"
                if mode not in splits:
                    continue

            else:
                raise ValueError(
                    f"MODE must be one of [train, val, test], you gave me {hparams.MODE} though..."
                )

            # hparams.SAM_MODE is either "sam" or "sam-sp"
            id_path = os.path.join(hparams.DATASET_ROOT, f, mode, f"{hparams.SAM_MODE}/ind")
            od_path = os.path.join(hparams.DATASET_ROOT, f, mode, f"{hparams.SAM_MODE}/ood")

            if 0 not in hparams.EXCLUDED_LABELS:
                for vec in os.listdir(id_path):
                    vec_path = os.path.join(id_path, vec)
                    v = np.load(vec_path)
                    assert len(v.shape) == 2
                    self.features.extend([v])
                    self.labels.extend([0] * v.shape[0])
            if 1 not in hparams.EXCLUDED_LABELS and os.path.exists(od_path):
                for vec in os.listdir(od_path):
                    vec_path = os.path.join(od_path, vec)
                    v = np.load(vec_path)
                    assert len(v.shape) == 2
                    self.features.extend([v])
                    self.labels.extend([1] * v.shape[0])
                    ood_counts.extend([v.shape[0]])

        self.ood_counts = ood_counts
        self.features = np.concatenate(self.features)

        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        # if SHAPE_MODE is a list, then it represents a bitmask of which token features to use
        # 1 denotes that the token is included, 0 denotes that it is excluded
        if isinstance(self.hparams.SHAPE_MODE, list):
            assert len(self.hparams.SHAPE_MODE) == 8

            shape = len(feature)
            token_len = 256 # based on SAM hyperparameters
            # the SAM feature is of shape 2048, each 256 block represents a token
            # we need to eliminate the blocks whose bitmask is equal to 0 and concatenate the rest

            feature = np.concatenate(
                [
                    feature[i * token_len : (i + 1) * token_len]
                    for i, bit in enumerate(self.hparams.SHAPE_MODE)
                    if bit == 1
                ],
                axis=0,
            )
    
        elif self.hparams.SHAPE_MODE == '2D':
            shape = len(feature)
            assert shape % 8 == 0, "it is assumed that SAM feature size is divisble by 8"
            # feature = feature.reshape(-1, 8)
            feature = feature.reshape(-1, 256).transpose(1, 0)

        return torch.from_numpy(feature), label


class LCLabels(Dataset):
    def __init__(self, hparams):
        self.hparams = hparams
        folders = os.listdir(hparams.DATASET_ROOT)

        self.labels = []

        for f in folders:
            if f in hparams.EXCLUDED_FOLDERS:
                continue

            splits = os.listdir(os.path.join(hparams.DATASET_ROOT, f))

            if hparams.MODE == "train":
                mode = "train"
                if "sam" in splits:
                    mode = ""

            elif hparams.MODE == "val":
                mode = "val"
                if mode not in splits:
                    continue

            elif hparams.MODE == "test":
                mode = "test"
                if mode not in splits:
                    continue

            path = os.path.join(hparams.DATASET_ROOT, f, mode, "labels")

            for l in os.listdir(path):
                if ".png" not in l:
                    continue
                self.labels.extend([os.path.join(path, l)])

        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        lbl = read_label(self.labels[index])

        return torch.from_numpy(lbl)
