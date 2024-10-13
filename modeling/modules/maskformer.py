import torch.nn as nn
from .misc import create_model
from easydict import EasyDict as edict
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder



DECODERS = edict(
    transformer_decoder=MultiScaleMaskedTransformerDecoder,
    pixel_decoder=MSDeformAttnPixelDecoder,
)

class MaskFormerHead(nn.Module):

    def __init__(
        self,
        num_classes: int,
        transformer_predictor_config: dict,
        pixel_decoder_config: dict,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        
        
        self.pixel_decoder = create_model(pixel_decoder_config, DECODERS)
        self.predictor = create_model(transformer_predictor_config, DECODERS)
        self.num_classes = num_classes


    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        
        predictions = self.predictor(multi_scale_features, mask_features, mask)
    
        return predictions
    

class MaskHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_classes,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        

    def forward(self, x, **kwargs):
        return x