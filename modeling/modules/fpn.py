import torch.nn as nn
from .misc import Upsample, resize
from easydict import EasyDict as edict

class FPN(nn.Module):

    def __init__(
            self,
            in_channels,
            output_dim,
            feature_names, # list
        ):
        super().__init__()
        self.feature_names = feature_names
        self.align_corners = False
        # FPN
        assert len(in_channels) == 4, "FPN expects 4 intermediate features from the backbone"
        
        self.fpns = nn.ModuleList([
            nn.GroupNorm(1, in_channels[0]), # H / 4,
            nn.GroupNorm(1, in_channels[1]), # H / 8,
            nn.GroupNorm(1, in_channels[2]), # H / 16,
            nn.GroupNorm(1, in_channels[3])  # H / 32,
        ])
       
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

        # fpn 1
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

        # fpn 2
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
        # fpn 3
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

        # fpn 4
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

    def get_output_features(self, interm_features):
        """
        image_input: last layer output feature of the backbone (B, N, C)
        interm_features: intermediate features from the backbone (4 x (B, C, H, W))
        
        NOTE: C here is the embedding size that is output by the backbone
        """

        features = []
        for i, name in enumerate(self.feature_names):
            features.append(self.fpns[i](interm_features[name]))  # (B, C, H, W)->  (B, C, kH, kW)

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

    def forward(self, interm_features, **kwargs):

        output = self.get_output_features(interm_features)

        return output



SEGMENTATION_DECODERS = edict(
    FPN=FPN,
)