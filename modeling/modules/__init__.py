
from .encoders import ENCODERS
from .decoders import DECODERS

from .dinov2 import DINOv2
from .gmm_seg import GMMSegHead
from .fpn import FPN
from .swin import SwinTransformer

from .misc import (
    ResidualLayer,
    ResidualStack,
    MLP,
    MultiScaleMLP,
    CascadedMLP,
    LinearHead,
    create_model
)

from .maskformer import MaskHead


from .vq_vae import VQVAE
from .pixel_cnn import GatedPixelCNN
from .mingpt_adapted import GPT


from easydict import EasyDict as edict

MODELS = edict(
    VQ_VAE=VQVAE,
    PixelCNN=GatedPixelCNN,
    GPT=GPT,
    DINOv2=DINOv2,
    GMMSegHead=GMMSegHead,
    SwinTransformer=SwinTransformer,
    LinearHead=LinearHead,
    MaskHead=MaskHead,
)

BACKBONE_OUTPUT_DIMS = edict(
    dinov2_vits14_reg=384,
    dinov2_vitb14_reg=768,
    dinov2_vitl14_reg=1024,
    dinov2_vitg14_reg=1536
)