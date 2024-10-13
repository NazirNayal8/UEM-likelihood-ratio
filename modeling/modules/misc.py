import inspect
import torch.nn as nn
import warnings
import torch
import torch.nn.functional as F
from torch import Tensor
from easydict import EasyDict as edict

def create_model(config, MODEL_DICT):

    MODEL = MODEL_DICT[config.NAME]

    params = edict()
    # gets the names of variables expected as input
    expected_params = inspect.getfullargspec(MODEL).args

    for k, v in config.items():
        # params in config are expected to be exact uppercase versions of param name
        p = k.lower()
        if p in expected_params:
            params[p] = v

    return MODEL(**params)



class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_hiddens: int):
        super(ResidualLayer, self).__init__()
        # NOTE: In reference code a ReLU is added before first CONV
        self.resblock = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, num_hiddens, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(num_hiddens, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)
    

class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                ResidualLayer(in_channels, hidden_dim, residual_hidden_dim)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class LinearHead(nn.Module):

    def __init__(self, embedding_dim, num_classes):
        super(LinearHead, self).__init__()
        self._head = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, **kwargs):
        """
        Expects input of shape (B, C, H, W)

        Returns output of shape (B, num_classes, H, W)
        NOTE: this interface conforms with that of GMMSegHead
        """
        
        x = x.permute(0, 2, 3, 1)
        x = self._head(x)
        x = x.permute(0, 3, 1, 2)

        return edict(sem_seg=x)

        

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self._layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ]
            + [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
            * (num_layers - 2)
            + [nn.Linear(hidden_dim, output_dim)]
        )

    def forward(self, x, **kwargs):
        
        for layer in self._layers:
            x = layer(x)
        return x
    

class MultiScaleMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_scales):
        super(MultiScaleMLP, self).__init__()
        self._mlps = nn.ModuleList(
            [
                MLP(input_dim, hidden_dim, output_dim, num_layers)
                for _ in range(num_scales)
            ]
        )

    def forward(self, multiscale_features, **kwargs):
        """
        Fuses the multiscale by addition
        """

        return sum([mlp(x) for x, mlp in zip(multiscale_features, self._mlps)])
        

class CascadedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_scales):
        super(CascadedMLP, self).__init__()
        self._mlps = nn.ModuleList()
        previous_output_dim = 0
        for _ in range(num_scales):
            mlp = MLP(input_dim + previous_output_dim, hidden_dim, output_dim, num_layers)
            self._mlps.append(mlp)
            previous_output_dim = output_dim


    def forward(self, multiscale_features, **kwargs):
        """
        Concatenates output of the previous layer with the input of the next one,
        each feature is of shape (B, N, C)
        """
        
        x = self._mlps[0](multiscale_features[0])
        for mlp, features in zip(self._mlps[1:], multiscale_features[1:]):
            x = torch.cat([x, features], dim=-1)
            x = mlp(x)
            
        return x

def l2_normalize(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)