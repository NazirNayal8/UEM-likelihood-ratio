import torch.nn as nn
import torch

from .misc import MLP, Upsample, resize
from .maskformer import MaskFormerHead


class FlatToMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, H, W):
        """
        (B, N, C) -> (B, C, H, W)
        """
        B, N, C = x.shape
        assert H * W == N, "H * W must be equal to N"
        return x.view(B, H, W, C).permute(0, 3, 1, 2)


class MapToFlat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        (B, C, H, W) -> (B, N, C)
        """
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).view(B, -1, C)


class DINOv2ConvUpsampling(nn.Module):

    def __init__(
        self,
        input_dim,
        num_pre_linear_layers,
        pre_hidden_dim,
        num_conv_layers,
        conv_hidden_dim,
        num_post_linear_layers,
        post_hidden_dim,
        output_dim,
    ):
        super().__init__()

        self.num_conv_layer = num_conv_layers

        self.pre_linear_layers = None
        self.conv_layers = None
        self.post_linear_layers = None

        if num_pre_linear_layers > 0:
            self.pre_linear_layers = MLP(
                input_dim, pre_hidden_dim, pre_hidden_dim, num_pre_linear_layers
            )

        if num_conv_layers > 0:
            conv_layers = []
            previous_layer_dim = (
                pre_hidden_dim if num_pre_linear_layers > 0 else input_dim
            )
            for i in range(num_conv_layers):
                conv_layers.extend(
                    [
                        nn.ConvTranspose2d(
                            previous_layer_dim,
                            conv_hidden_dim,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                        nn.ReLU(),
                    ]
                )
                previous_layer_dim = conv_hidden_dim
            if num_post_linear_layers == 0:
                conv_layers.append(
                    nn.Conv2d(
                        previous_layer_dim,
                        output_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
            self.conv_layers = nn.Sequential(*conv_layers)

        if num_post_linear_layers > 0:
            self.post_linear_layers = MLP(
                conv_hidden_dim, post_hidden_dim, output_dim, num_post_linear_layers
            )

    def forward(self, x, H, W, intermediate_features=None):

        B, N, C = x.shape

        if self.pre_linear_layers is not None:
            x = self.pre_linear_layers(x)

        if self.conv_layers is not None:
            x = FlatToMap()(x, H, W)
            x = self.conv_layers(x)
            upscale_factor = 2**self.num_conv_layer
            H, W = H * upscale_factor, W * upscale_factor

        if self.post_linear_layers is not None:
            x = MapToFlat()(x)
            x = self.post_linear_layers(x)
            x = FlatToMap()(x, H, W)

        return x


class DINOv2FPN(nn.Module):

    def __init__(
        self,
        embed_dim,
        in_channels,
        output_dim,
    ):
        super().__init__()

        self.align_corners = False
        # FPN
        self.fpn1 = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.GroupNorm(1, embed_dim)

        self.fpn4 = nn.Sequential(
            nn.GroupNorm(1, embed_dim), nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dino_mlp = MLP(
            input_dim=embed_dim, 
            hidden_dim=512, 
            output_dim=output_dim, 
            num_layers=2
        )

        # neck
        # NOTE: the emebd_dim + output_dim is because the output feature from the backbone is
        # concatenated with the third fpn layer
        in_channels = [embed_dim, embed_dim, embed_dim + output_dim, embed_dim]
        self.in_channels = in_channels
        self.output_dim = output_dim  # 256
        self.num_ins = len(in_channels)
        self.num_outs = len(in_channels)

        self.lateral_convs = nn.ModuleList()
        self.lateral_convs.append(
            nn.Conv2d(self.in_channels[0], self.output_dim, 1, stride=1)
        )
        self.lateral_convs.append(
            nn.Conv2d(self.in_channels[1], self.output_dim, 1, stride=1)
        )
        self.lateral_convs.append(
            nn.Conv2d(self.in_channels[2], self.output_dim, 1, stride=1)
        )
        self.lateral_convs.append(
            nn.Conv2d(self.in_channels[3], self.output_dim, 1, stride=1)
        )

        self.fpn_convs = nn.ModuleList()
        self.fpn_convs.append(
            nn.Conv2d(self.output_dim, self.output_dim, 3, stride=1, padding=(1, 1))
        )
        self.fpn_convs.append(
            nn.Conv2d(self.output_dim, self.output_dim, 3, stride=1, padding=(1, 1))
        )
        self.fpn_convs.append(
            nn.Conv2d(self.output_dim, self.output_dim, 3, stride=1, padding=(1, 1))
        )
        self.fpn_convs.append(
            nn.Conv2d(self.output_dim, self.output_dim, 3, stride=1, padding=(1, 1))
        )

        self.heads = nn.ModuleList()
        self.heads.append(
            nn.Sequential(
                nn.Conv2d(
                    self.output_dim,
                    self.output_dim,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.SyncBatchNorm(self.output_dim),
                nn.ReLU(inplace=True),
            )
        )

        self.heads.append(
            nn.Sequential(
                nn.Conv2d(
                    self.output_dim,
                    self.output_dim,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.SyncBatchNorm(self.output_dim),
                nn.ReLU(inplace=True),
                Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
            )
        )

        self.heads.append(
            nn.Sequential(
                nn.Conv2d(
                    self.output_dim,
                    self.output_dim,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.SyncBatchNorm(self.output_dim),
                nn.ReLU(inplace=True),
                Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
                nn.Conv2d(
                    self.output_dim,
                    self.output_dim,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.SyncBatchNorm(self.output_dim),
                nn.ReLU(inplace=True),
                Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
            )
        )

        self.heads.append(
            nn.Sequential(
                nn.Conv2d(
                    self.output_dim,
                    self.output_dim,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.SyncBatchNorm(self.output_dim),
                nn.ReLU(inplace=True),
                Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
                nn.Conv2d(
                    self.output_dim,
                    self.output_dim,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.SyncBatchNorm(self.output_dim),
                nn.ReLU(inplace=True),
                Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
                nn.Conv2d(
                    self.output_dim,
                    self.output_dim,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.SyncBatchNorm(self.output_dim),
                nn.ReLU(inplace=True),
                Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
            )
        )

    def get_output_features(self, image_input, H, W, interm_features):
        """
        image_input: last layer output feature of the backbone (B, N, C)
        interm_features: intermediate features from the backbone (4 x (B, C, H, W))
        
        NOTE: C here is the embedding size that is output by the backbone
        """

        x = self.dino_mlp(image_input) # -> (B, N, self.output_dim)
        x = FlatToMap()(x, H, W) # -> (B, self.output_dim, H, W)

        features = []
        features.append(self.fpn1(interm_features[0]))  # (B, C, H, W)->  (B, C, 4H, 4W)
        features.append(self.fpn2(interm_features[1]))  # (B, C, H, W)->  (B, C, 2H, 2W)
        features.append(self.fpn3(interm_features[2]))  # (B, C, H, W)->  (B, C, H, W)
        features.append(self.fpn4(interm_features[3]))  # (B, C, H, W)->  (B, C, H/2, W/2)

        features[2] = torch.cat((features[2], x), dim=1) # -> (B, C + self.output_dim, H, W)
        # build laterals
        laterals = [
            lateral_conv(features[i]) # -> (B, self.output_dim, H, W) NOTE: H,W here is the same as the input, differs among laterals
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # prefix sum of laterals
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(laterals[i], size=prev_shape)

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)] # -> 4 x (B, self.output_dim, H, W)

        # combine
        output = self.heads[0](outs[0])
        for i in range(1, 4):
            # non inplace
            output = output + resize(
                self.heads[i](outs[i]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=True,
            ) # -> (B, self.output_dim, H, W)

        return output

    def forward(self, x, intermediate_features, H, W, inference_mode=False):

        intermediate_features = [
            FlatToMap()(interm_feature, H, W)
            for interm_feature in intermediate_features # -> 4 x (B, C, H, W)
        ]

        output = self.get_output_features(x, H, W, intermediate_features)

        return output


class DINOv2(nn.Module):

    def __init__(self, version, features_type, freeze, interm_features, learnable_params):
        super().__init__()
        self.version = version
        self.features_type = features_type
        self.learnable_params = learnable_params
        self.interm_features = interm_features
        self.dinov2 = torch.hub.load(
            "facebookresearch/dinov2", version, pretrained=True
        )
        if freeze:
            for param in self.dinov2.parameters():
                param.requires_grad = False

        if learnable_params is not None:
            if learnable_params.TYPE == "mlp":
                self.learnable_layers = MLP(
                    input_dim=self.dinov2.embed_dim,
                    hidden_dim=learnable_params.HIDDEN_DIM,
                    output_dim=learnable_params.OUTPUT_DIM,
                    num_layers=learnable_params.NUM_LAYERS,
                )

            elif learnable_params.TYPE == "conv_upsampling":
                self.learnable_layers = DINOv2ConvUpsampling(
                    input_dim=self.dinov2.embed_dim,
                    num_pre_linear_layers=learnable_params.NUM_PRE_LINEAR_LAYERS,
                    pre_hidden_dim=learnable_params.PRE_HIDDEN_DIM,
                    num_conv_layers=learnable_params.NUM_CONV_LAYERS,
                    conv_hidden_dim=learnable_params.CONV_HIDDEN_DIM,
                    num_post_linear_layers=learnable_params.NUM_POST_LINEAR_LAYERS,
                    post_hidden_dim=learnable_params.POST_HIDDEN_DIM,
                    output_dim=learnable_params.OUTPUT_DIM,
                )
            elif learnable_params.TYPE == "fpn":
                self.learnable_layers = DINOv2FPN(
                    embed_dim=self.dinov2.embed_dim,
                    in_channels=learnable_params.IN_CHANNELS,
                    output_dim=learnable_params.OUTPUT_DIM,
                )
            elif learnable_params.TYPE == "maskformer":
                self.learnable_layers = MaskFormerHead(
                    num_classes=learnable_params.NUM_CLASSES,
                    transformer_predictor_config=learnable_params.TRANSFORMER_PREDICTOR_CONFIG,
                    pixel_decoder_config=learnable_params.PIXEL_DECODER_CONFIG,
                )

    
    def get_output_and_interm_features(self, image_input):

        output = self.dinov2.forward_features(image_input)[self.features_type]

        if self.interm_features is not None:
            interm_features = self.dinov2.get_intermediate_layers(image_input, n=self.interm_features)
            return output, interm_features
        else:
            return output, None

    def forward(self, x, return_dino_features=False):
        """
        Input:
        - x: input image tensor (B, C, H, W)

        NOTE: H and W must be divisible by 14
        """
        B, C, H, W = x.shape
        H = H // self.dinov2.patch_size
        W = W // self.dinov2.patch_size

        with torch.no_grad():
            # output = self.dinov2.forward_features(x)[self.features_type]
            dino_features = self.get_output_and_interm_features(x)
            output, intermediate_features = dino_features # (B, N, C), 4 x (B, C, H, W)

        if self.learnable_params is not None:
            if self.learnable_params.TYPE == "maskformer":
                inputs = {}
                for i, idx in enumerate(self.interm_features):
                    inputs[f"res{idx}"] = intermediate_features[i].reshape(B, H, W, -1).permute(0, 3, 1, 2)
                    
                output = self.learnable_layers(inputs)
                
            else:
                output = self.learnable_layers(output, H=H, W=W, intermediate_features=intermediate_features)
            if self.learnable_params.TYPE == "mlp":
                output = FlatToMap()(output, H, W)
        else:
            C = output.size(2)  # hidden size
            output = output.view(B, H, W, C).permute(0, 3, 1, 2)
        
        if return_dino_features:
            return output, dino_features

        return output
