import torch.nn as nn
import torch.nn.functional as F

from .misc import ResidualStack
from easydict import EasyDict as edict


class MLPDecoder(nn.Module):

    def __init__(
            self,
            num_tokens,
            in_channels,
            hidden_sizes,
            out_channels,
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        
        in_channels = num_tokens * in_channels
        hidden_sizes = [in_channels] + hidden_sizes
        
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Expected quantized input from codebook of shape [batch_size, num_channels, num tokens]
        """
        B, C, N = inputs.shape
        inputs = inputs.reshape(B, C * N)
        x = self.layers(inputs)
        return x

class DecoderConv(nn.Module):
    """ 
    Args:
        in_channels: Number of channels in the input tensor.
        out_channels: Number of channels in the output tensor.
        hidden_dim: Number of channels in the hidden activations.
        num_residual_layers: Number of residual layers in the decoder.
        residual_hidden_dim: Number of channels in the residual hidden
            activations.
        attention: Whether to use attention in the decoder.
        attention_params: Parameters for MultiheadAttention module.
    Returns:
        A torch.Tensor of shape [batch_size, out_channels, height * 2,
            width * 2].
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_residual_layers,
        residual_hidden_dim,
        upsample
    ):
        super().__init__()

        if upsample:
            upsample_conv =  nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            upsample_conv =  nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualStack(
                in_channels=hidden_dim,
                hidden_dim=hidden_dim,
                num_residual_layers=num_residual_layers,
                residual_hidden_dim=residual_hidden_dim,
            ),
            upsample_conv,
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim // 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, inputs):
        
        x = self.layers(inputs)

        return x


class FeatureDecoder(nn.Module):
    """
    Decodes the output of FeatureEncoder class to the original image dimensions.

    Args:
        in_channels: Number of channels in the input tensor.
        out_channels: Number of channels in the output tensor.
        hidden_dim: Number of channels in the hidden activations.
        num_residual_layers: Number of residual layers in the decoder.
        residual_hidden_dim: Number of channels in the residual hidden
            activations.

    Returns:
        A torch.Tensor of shape [batch_size, out_channels, height * 2,
            width * 2].
    
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_residual_layers,
        residual_hidden_dim,
    ):
        super().__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._residual_stack = ResidualStack(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim,
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        return self.deconv_layers(x)


class DecoderFlex(nn.Module):
    """
    A Decoder network for VQ-VAE with flexibe hidden dimension sizes.

    Args:
        in_channels: Number of channels in the input tensor.
        hidden_dim: Number of channels in the hidden activations.
        num_residual_layers: Number of residual layers in the decoder.
        residual_hidden_dim: Number of channels in the residual hidden
            activations.
        attention: Whether to use attention in the decoder.
        attention_params: Parameters for MultiheadAttention module.
    
    Returns:
        A torch.Tensor of shape [batch_size, 3, height * 2, width * 2].

    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_residual_layers,
        residual_hidden_dim,
        attention=False,
        attention_params=None,
    ):
        super(DecoderFlex, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._residual_stack = ResidualStack(
            in_channels=hidden_dim[2],
            hidden_dim=hidden_dim[2],
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim,
        )

        self._conv_trans_0 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dim[2],
                out_channels=hidden_dim[1],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=hidden_dim[1],
            out_channels=hidden_dim[0],
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=hidden_dim[0],
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.attention = attention

        if attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim[2],
                num_heads=attention_params.NUM_HEADS,
                dropout=attention_params.DROPOUT,
                batch_first=True,
            )

    def forward(self, inputs):
        x = self._conv_1(inputs)

        # shape of x is: [batch_size, hidden_dim, height, width]

        if self.attention:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).view(B, H * W, C)
            x = self.attention_layer(x, x, x)[0]
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self._residual_stack(x)

        x = self._conv_trans_0(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)

DECODERS = edict(
    DecoderConv =DecoderConv,
    MLPDecoder=MLPDecoder,
    FEATURE_DECODER=FeatureDecoder,
    DECODER_FLEX=DecoderFlex
)