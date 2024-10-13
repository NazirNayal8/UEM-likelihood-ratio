import numpy as np
import torch
import yaml


def read_config(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def overwrite_base_config(base_config, config):

    for k, v in config.items():
        if k == "BASE_CONFIG":
            continue
        if k not in base_config:
            base_config[k] = v
        elif isinstance(v, dict):
            base_config[k] = overwrite_base_config(base_config[k], config[k])
        else:
            base_config[k] = v

    return base_config

def read_config_recursive(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
        if "BASE_CONFIG" in config:
            base_config = read_config_recursive(config["BASE_CONFIG"])
            config = overwrite_base_config(base_config, config)

    return config


def update_from_wandb(config, wandb_config):

    for k, v in wandb_config.items():
        if k not in config:
            raise ValueError(f"Wandb Config has sth that you don't: {k}")
        if isinstance(v, dict):
            config[k] = update_from_wandb(config[k], wandb_config[k])
        else:
            config[k] = v

    return config


# overwrite config with opts from args, each argument can be a string
# that can contain . to access nested dicts. opts is a list of 
# even number of elements showing the argument and its value
def overwrite_config(config, opts):

    if opts is None:
        return config

    assert len(opts) % 2 == 0, "opts should be a list of even number of elements"

    for i in range(0, len(opts), 2):
        arg = opts[i]
        val = yaml.safe_load(opts[i + 1]) # for automatic casting
        # val is a string, but infer its type and cast it
        if "." in arg:
            keys = arg.split(".")
            c = config
            for k in keys[:-1]:
                c = c[k]
            c[keys[-1]] = val
        else:
            config[arg] = val

    return config



# get two numpy masks and compute intersection over union between them
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


# take a list of vectors and another for their labels and plot tsne
def plot_tsne(vectors, labels, title):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 10))
    # choose a discriminative cmap
    ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap="Paired")
    ax.set_title(title)
    plt.show()


# take a list of vectors and another for their labels and plot UMAP
def plot_umap(vectors, labels, title):
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP()
    vectors_2d = reducer.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 10))
    # choose a discriminative cmap
    ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap="Paired")
    ax.set_title(title)
    plt.show()

def get_connected_components(mask, filter_pixel_num=None):
    from skimage import measure
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    # find connected components
    labels = measure.label(mask, connectivity=2)
    # get the number of connected components
    num_components = np.max(labels) + 1
    # get the connected components
    components = []
    empty_mask = (mask == 0)
    for i in range(num_components):
        m = (labels == i)
        if compute_iou(m, empty_mask) > 0:
            continue
        if filter_pixel_num is not None:
            if np.sum(m) < filter_pixel_num:
                continue
        components.append(m)
    return components

