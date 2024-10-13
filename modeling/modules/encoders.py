import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .misc import ResidualStack
from easydict import EasyDict as edict


class EncoderFlexibleTokenizing(nn.Module):
    """
    An encoder that takes an input vector and outputs a set of vectors where
    the number of these vectors is an input paremters.
    """

    def __init__(self, in_channels, hidden_sizes, out_channels, num_tokens):
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_sizes = [in_channels] + hidden_sizes
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], out_channels * num_tokens))
        self.layers = nn.Sequential(*layers)

        self.out_channels = out_channels
        self.num_tokens = num_tokens
    
    def forward(self, x):
        """
        Expected input of shape [batch_size, in_channels]
        """
        x = self.layers(x)
        x = x.reshape(x.shape[0], self.out_channels, self.num_tokens)
        return x
    
class EncoderLinearLight(nn.Module):

    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        layers = [nn.Linear(in_channels, out_channels)]
        
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.pooling = nn.AvgPool1d(2, 2)

        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, inputs):
        """
        Expected input of shape [batch_size, in_channels, seq_len]
        """
        inputs = inputs.permute(0, 2, 1)
        x = self.layers(inputs)
        x = x.permute(0, 2, 1)
        if self.downsample:
            x = self.pooling(x)
            
        return x

class EncoderConvLight(nn.Module):
    """
    A light convolutional encoder network for VQ-VAE with a single conv layer.
    
    """

    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        if downsample:
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        self.layers = nn.Sequential(
            conv_layer,
            nn.ReLU(),
        )

        self.out_channels = out_channels

    def forward(self, inputs):
        x = self.layers(inputs)
        return x
    

class EncoderConv(nn.Module):
    """
    A Convolutional Encoder network for VQ-VAE.

    Args:
        in_channels: Number of channels in the input tensor.
        hidden_dim: Number of channels in the hidden activations.
        num_residual_layers: Number of residual layers in the encoder.
        residual_hidden_dim: Number of channels in the residual hidden
            activations.

    Returns:
        A torch.Tensor of shape [batch_size, hidden_dim, height // 2,
            width // 2].
    """

    def __init__(
        self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_residual_layers = num_residual_layers
        self.residual_hidden_dim = residual_hidden_dim

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            ResidualStack(
                in_channels=hidden_dim,
                hidden_dim=hidden_dim,
                num_residual_layers=num_residual_layers,
                residual_hidden_dim=residual_hidden_dim,
            )
        )

        self.out_channels = self.get_out_channels()

    def get_out_channels(self):
        """
        Returns:
            out_channels: Number of output channels of the encoder
        """
        with torch.no_grad():
            dummy_input = torch.zeros((1, self.in_channels, 8), device=self.layers[0].weight.device)
            output = self.layers(dummy_input)
            
        return output.shape[1]

    def forward(self, inputs):
        
        x = self.layers(inputs)
        
        return x


class FeatureEncoder(nn.Module):
    """
    A class that takes a feature map of dimensions [H, W, in_channels], processes it using convolutional 
    networks so that it becomes a map of dimension [H//4, W//4, hidden_dim], using convolutional networks, ReLU 
    activation functions and residual blocks.
    """

    def __init__(
        self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        self._residual_stack = ResidualStack(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim,
        )

    def forward(self, inputs):
        x = self.conv_layers(inputs)

        x = self._residual_stack(x)

        return x


class EncoderDINOV2(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        assert hparams.FEATURES_TYPE in [
            "x_norm_clstoken",
            "x_prenorm",
            "x_norm_patchtokens",
        ]

        if hparams.NAME == "dinov2_vits14":
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2", hparams.NAME, pretrained=True
            )
        else:
            raise ValueError(f"Unknown model name: {hparams.NAME}")

        self.avgpool = nn.AvgPool1d(
            kernel_size=hparams.KERNEL_SIZE, stride=hparams.STRIDE
        )

        # freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [batch_size, channels, height, width]

        Returns:
            Tensor of shape []
        """
        with torch.no_grad():
            x = self.encoder.forward_features(inputs)[self.hparams.FEATURES_TYPE]
            if self.hparams.FEATURES_TYPE in ["x_prenorm", "x_norm_patchtokens"]:
                x = self.avgpool(x.permute(0, 2, 1))
                x = x.permute(0, 2, 1)
        B, N, C = x.shape
        N_sqrt = int(np.sqrt(N))
        assert N_sqrt * N_sqrt == N, "Num of patches must be a perfect square"

        x = x.view(B, N_sqrt, N_sqrt, C).permute(0, 3, 1, 2)

        return x


class EncoderFlex(nn.Module):
    """
    Encoder for VQ-VAE with flexible number of layers

    Args:
        in_channels (int): Number of input channels
        hidden_dim (list): List of hidden dimensions
        num_residual_layers (int): Number of residual layers
        residual_hidden_dim (int): Number of hidden dimensions for residual layers
    """

    def __init__(
        self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim
    ):
        super(EncoderFlex, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim[0],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=hidden_dim[0],
            out_channels=hidden_dim[1],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=hidden_dim[1],
            out_channels=hidden_dim[2],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=hidden_dim[2],
            hidden_dim=hidden_dim[2],
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)

ENCODERS = edict(
    EncoderConv=EncoderConv,
    EncoderConvLight=EncoderConvLight,
    EncoderLinearLight=EncoderLinearLight,
    EncoderFlexibleTokenizing=EncoderFlexibleTokenizing,
    FEATURE_ENCODER=FeatureEncoder,
    ENCODER_DINOV2=EncoderDINOV2,
    ENCODER_FLEX=EncoderFlex
)