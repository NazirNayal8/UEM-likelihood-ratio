import torch
import numpy as np
import matplotlib.pyplot as plt

from easydict import EasyDict as edict


def proc_img(img):

    if isinstance(img, torch.Tensor):
        ready_img = img.clone()
        if len(ready_img.shape) == 3 and ready_img.shape[0] == 3:
            ready_img = ready_img.permute(1, 2, 0)
        ready_img = ready_img.cpu()

    elif isinstance(img, np.ndarray):
        ready_img = img.copy()
        if len(ready_img.shape) == 3 and ready_img.shape[0] == 3:
            ready_img = ready_img.transpose(1, 2, 0)
    else:
        raise ValueError(
            f"Unsupported type for image: ({type(img)}), only supports numpy arrays and Pytorch Tensors"
        )

    return ready_img


def show_quals(
    images,
    structure,
    text,
    img_shape,
    text_loc=edict(x=25, y=700, f=42),
    save_name=None,
    circles=None,
    column_title=False,
    wspace=0,
    hspace=0,
    cmap="viridis",
    file_format="pdf",
):
    num_cols = len(structure[0])
    num_rows = len(structure)

    width = img_shape[1] * num_cols / 100
    height = width * img_shape[0] * num_rows / (img_shape[1] * num_cols)

    fig = plt.figure(constrained_layout=True, figsize=(width, height))
    ax = fig.subplot_mosaic(
        structure,
        gridspec_kw={
            "bottom": 0.0,
            "top": 1.0,
            "left": 0.0,
            "right": 1.0,
            "wspace": wspace,
            "hspace": hspace,
            "height_ratios": [1] * num_rows,
            "width_ratios": [1] * num_cols,
        },
    )

    for i, (k, v) in enumerate(images.items()):
        ax[k].imshow(proc_img(v), cmap=cmap)
        ax[k].axis("off")
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        t = ""
        for n, m in text.items():
            if n in k:
                t = m
        if column_title:
            if k in structure[0]:
                ax[k].text(
                    x=text_loc.x,
                    y=text_loc.y,
                    s=t,
                    fontsize=text_loc.f,
                    fontweight="extra bold",
                    color="black",
                    horizontalalignment="center",
                )
        else:
            ax[k].text(
                x=text_loc.x,
                y=text_loc.y,
                s=t,
                fontsize=text_loc.f,
                fontweight="extra bold",
                color="white",
            )

        if circles is not None:
            for n, p in circles.items():
                if n in k:
                    c = plt.Circle(
                        (p.x, p.y), p.r, fill=False, color="white", linewidth=p.lw
                    )
                    ax[k].set_aspect(1)
                    ax[k].add_artist(c)

    plt.margins(0, 0)
    fig.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0, left=0, right=0, bottom=0, top=0)
    if save_name is not None:
        plt.savefig(
            f"plots/{save_name}.{file_format}",
            format=file_format,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()
    plt.close(fig)
