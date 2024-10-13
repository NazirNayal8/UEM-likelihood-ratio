import matplotlib.pyplot as plt

def unnormalize_image(img, mean, std):
    """
    img is a tensor and mean and std are lists of length 3
    
    return a channel-wise unnormalize image with the given mean and std
    return as torch tensor
    """
    import torch
    mean = [m * 255 for m in mean]
    std = [s * 255 for s in std]
    img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    img = img.permute(1, 2, 0).clamp(0, 255).numpy().astype('uint8')
    return img


def plot_side_by_side(img1, img2, path, title=None):
    # increase figure size of plotted images
    
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[0].axis('off')
    ax[1].axis('off')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()

# plot 3 images side by side the same as the previous function
def plot_side_by_side_3(img1, img2, img3, path, title=None):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(20, 10)
    
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[2].imshow(img3)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()