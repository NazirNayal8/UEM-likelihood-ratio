import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
from .segmentation import SegmentationModel
from .modules import (
    create_model,
    MLP,
    MultiScaleMLP,
    CascadedMLP,
    MODELS,
    BACKBONE_OUTPUT_DIMS,
)

from torchmetrics import JaccardIndex, AUROC, AveragePrecision, MetricCollection
from modeling.modules.metrics import FPR95
from easydict import EasyDict as edict

from .schedulers.poly import get_polynomial_decay_schedule_with_warmup


class OoDSegmentationModel(pl.LightningModule):

    def __init__(self, config, segmentor_ckpt=None):
        super().__init__()

        self.save_hyperparameters(config)

        if segmentor_ckpt is None:
            self.segmentor = SegmentationModel.load_from_checkpoint(
                self.hparams.MODEL.SEGMENTOR_CKPT
            )
        else:
            self.segmentor = SegmentationModel.load_from_checkpoint(
                segmentor_ckpt)
        for param in self.segmentor.parameters():
            param.requires_grad = False

        if self.hparams.MODEL.get("OOD_PROJECTOR", None) is None:
            self.ood_projector = None
        elif self.hparams.MODEL.OOD_PROJECTOR.TYPE == "mlp":
            self.ood_projector = MLP(
                input_dim=self.hparams.MODEL.OOD_PROJECTOR.INPUT_DIM,
                hidden_dim=self.hparams.MODEL.OOD_PROJECTOR.HIDDEN_DIM,
                output_dim=self.hparams.MODEL.OOD_PROJECTOR.OUTPUT_DIM,
                num_layers=self.hparams.MODEL.OOD_PROJECTOR.NUM_LAYERS,
            )
        elif self.hparams.MODEL.OOD_PROJECTOR.TYPE == "mlp_multiscale":
            self.ood_projector = MultiScaleMLP(
                input_dim=self.hparams.MODEL.OOD_PROJECTOR.INPUT_DIM,
                hidden_dim=self.hparams.MODEL.OOD_PROJECTOR.HIDDEN_DIM,
                output_dim=self.hparams.MODEL.OOD_PROJECTOR.OUTPUT_DIM,
                num_layers=self.hparams.MODEL.OOD_PROJECTOR.NUM_LAYERS,
                num_scales=self.hparams.MODEL.OOD_PROJECTOR.NUM_SCALES,
            )
        elif self.hparams.MODEL.OOD_PROJECTOR.TYPE == "mlp_cascaded":
            self.ood_projector = CascadedMLP(
                input_dim=self.hparams.MODEL.OOD_PROJECTOR.INPUT_DIM,
                hidden_dim=self.hparams.MODEL.OOD_PROJECTOR.HIDDEN_DIM,
                output_dim=self.hparams.MODEL.OOD_PROJECTOR.OUTPUT_DIM,
                num_layers=self.hparams.MODEL.OOD_PROJECTOR.NUM_LAYERS,
                num_scales=self.hparams.MODEL.OOD_PROJECTOR.NUM_SCALES,
            )
        else:
            raise NotImplementedError(
                f"Given OoD Projector not supported: {self.hparams.MODEL.OOD_PROJECTOR.TYPE}"
            )

        self.ood_head = create_model(self.hparams.MODEL.OOD_HEAD, MODELS)

        self.iou_metric = JaccardIndex(
            task="multiclass",
            num_classes=self.segmentor.hparams.MODEL.NUM_CLASSES,
            ignore_index=self.segmentor.hparams.MODEL.IGNORE_INDEX,
        )

        self.ood_metrics_idd = MetricCollection(
            {
                "AUROC": AUROC(
                    task="binary",
                    ignore_index=self.hparams.MODEL.IGNORE_INDEX,
                ),
                "AUPR": AveragePrecision(
                    task="binary",
                    ignore_index=self.hparams.MODEL.IGNORE_INDEX,
                ),
                "FPR95": FPR95(
                    ignore_index=self.hparams.MODEL.IGNORE_INDEX,
                ),
            },
            prefix="idd_",
        )

        self.ood_metrics_ood = self.ood_metrics_idd.clone(prefix="ood_")

        self.ood_metrics_llr = self.ood_metrics_idd.clone(prefix="llr_")

        if self.hparams.DATA.EVAL_DATASET == "ra_and_fslaf":
            import torch.nn as nn
            self.ood_metrics_idd = nn.ModuleList([
                self.ood_metrics_idd.clone(prefix="idd_1"),
                self.ood_metrics_idd.clone(prefix="idd_2"),
            ])
            self.ood_metrics_ood = nn.ModuleList([
                self.ood_metrics_ood.clone(prefix="ood_1"),
                self.ood_metrics_ood.clone(prefix="ood_2"),
            ])
            self.ood_metrics_llr = nn.ModuleList([
                self.ood_metrics_llr.clone(prefix="llr_1"),
                self.ood_metrics_llr.clone(prefix="llr_2"),
            ])

    @classmethod
    def embedding_dim_consistency(self, config, args):
        """
        Ensure embedding dim is consistent in the following cases
        - If OoD Head is taking input directly from backbone, then the input dim
            must match the backbone dimension
        - If the OoD head is taking input from the ood_projector (not None), then
            the input dim of the ood_projector must match the output dim of the
            backbone, and the embedding dim of the OoD Head must match the output dim
            of the ood_projector
        """
        segmentor_hparams = edict(
            torch.load(config.MODEL.SEGMENTOR_CKPT)["hyper_parameters"]
        )

        if config.MODEL.get("OOD_PROJECTOR", None) is not None:
            config.MODEL.OOD_PROJECTOR.INPUT_DIM = BACKBONE_OUTPUT_DIMS[
                segmentor_hparams.MODEL.BACKBONE.VERSION
            ]
            config.MODEL.OOD_HEAD.EMBEDDING_DIM = config.MODEL.OOD_PROJECTOR.OUTPUT_DIM
        else:
            config.MODEL.OOD_HEAD.EMBEDDING_DIM = BACKBONE_OUTPUT_DIMS[
                segmentor_hparams.MODEL.BACKBONE.VERSION
            ]

        return config

    @classmethod
    def distributed_consistency(self, config, args):
        """
        If training is distributed, make sure the paramter is set in the OoD Head as well
        """
        if args.distributed:
            config.MODEL.OOD_HEAD.DISTRIBUTED_TRAINING = True
        else:
            config.MODEL.OOD_HEAD.DISTRIBUTED_TRAINING = False

        return config

    @classmethod
    def apply_consistency(self, config, args):

        config = self.embedding_dim_consistency(config, args)
        config = self.distributed_consistency(config, args)

        return config

    def forward(self, x, gt_ood=None, return_seg=True, return_ood=True):

        assert (
            return_seg or return_ood
        ), "At least one of return_seg or return_ood should be True"

        result = edict()

        features, dino_features = self.segmentor.backbone(
            x, return_dino_features=True)

        if return_seg:
            seg_out = self.segmentor.segmentation_head(features)
            result.update(seg_out)
        if return_ood:
            patch_size = self.segmentor.backbone.dinov2.patch_size
            B, C, H, W = x.shape
            H, W = H // patch_size, W // patch_size
            if self.ood_projector is not None:
                # NOTE: consider a utilization of intermediate features as well
                final_features, intermediate_features = (
                    dino_features  # (B, N, C), 4 x (B, N, C)
                )

                if (
                    "multiscale" in self.hparams.MODEL.OOD_PROJECTOR.TYPE
                    or "cascaded" in self.hparams.MODEL.OOD_PROJECTOR.TYPE
                ):
                    # NOTE: here we replace the final intermediate feature with the final features in this version
                    # can be handled in a different way though; for example by adding an additional layer
                    multiscale_features = list(intermediate_features)[
                        : self.hparams.MODEL.OOD_PROJECTOR.NUM_SCALES - 1
                    ] + [final_features]
                    features = self.ood_projector(
                        multiscale_features=multiscale_features
                    )
                else:
                    features = self.ood_projector(final_features)
                # OoD head expects input in (B, C, H, W) format

                # TODO: this strictly assumed dinov2, change it to make it general
                # to do this we need to move patch_size one level higher as
                # an attribute of the backbone
                features = features.view(B, H, W, -1).permute(
                    0, 3, 1, 2
                )  # -> (B, C, H, W)
                ood_out = self.ood_head(features, gt_semantic_seg=gt_ood)
            else:
                features = (
                    dino_features[0].view(B, H, W, -1).permute(0, 3, 1, 2)
                )  # -> (B, C, H, W)
                ood_out = self.ood_head(features, gt_semantic_seg=gt_ood)

            ood_out.ood_score = ood_out.sem_seg
            del ood_out["sem_seg"]

            result.update(ood_out)

        return result

    def configure_optimizers(self):

        if self.hparams.SOLVER.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.SOLVER.LR,
                weight_decay=self.hparams.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(
                f"Given optimizer not supported: {self.hparams.SOLVER.OPTIMIZER}"
            )

        if self.hparams.SOLVER.LR_SCHEDULER.NAME == "PolyWithLinearWarmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.SOLVER.LR_SCHEDULER.NUM_WARMUP_STEPS,
                num_training_steps=self.hparams.SOLVER.MAX_STEPS,
                lr_end=self.hparams.SOLVER.LR_SCHEDULER.LR_END,
                power=self.hparams.SOLVER.LR_SCHEDULER.POWER,
                last_epoch=self.hparams.SOLVER.LR_SCHEDULER.LAST_EPOCH,
            )

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def likelihood_ratio_loss(self, outputs, targets):
        """
        Inputs:
            outputs: dict containing the output of the model,
                the expected entries are:
                    - sem_seg: inlier semantic segmentation logits of shape (B, C, H, W)
                    - ood_score: logits of the OoD GMM Head of shape (B, 2, H, W)
                    - contrast_logits (optional): logits for the contrastive loss
                    - contrast_targets (optional): contrastive loss targets
            targets: ood mask where:
                0: inlier
                1: OoD
                IGNORE_INDEX: ignore

        Currently, the loss is calculated as:
            - Binary Cross Entropy between likelihood ratio and ood mask, plus
            - CrossEntropyLoss between ood_score and targets, plus
            - CrossEntropyLoss between contrast_logits and contrast_targets in case OoD head
                is a GMM

        """

        # LLR = log(p(x|ood)) - log(p(x|inlier)) - maxlogit(segmentor)
        lr_score = (
            outputs.ood_score[:, 1]
            - outputs.sem_seg.max(dim=1)[0]
            - outputs.ood_score[:, 0]
        )

        mask = targets != self.hparams.MODEL.IGNORE_INDEX

        lr_loss = F.binary_cross_entropy_with_logits(
            lr_score[mask], targets[mask].float()
        )

        # GMM loss for when the GMM loss is used. Controlled by a weight parameter
        gmm_loss = F.cross_entropy(
            outputs.ood_score,
            targets.long(),
            ignore_index=self.hparams.MODEL.IGNORE_INDEX,
        )

        lr_loss = lr_loss + self.hparams.MODEL.LOSS.GMM_LOSS_WEIGHT * gmm_loss

        total_loss = lr_loss

        # NOTE: not all segmentation heads have contrast logits, therefore compute it
        # only if it is present. This is also related to the GMM loss
        if "contrast_logits" in outputs:
            contrast_loss = F.cross_entropy(
                outputs.contrast_logits,
                outputs.contrast_targets.long(),
                ignore_index=self.hparams.MODEL.IGNORE_INDEX,
            )
            total_loss = (
                total_loss
                + self.hparams.MODEL.LOSS.CONTRAST_LOSS_WEIGHT * contrast_loss
            )

        return total_loss

    def loss_function(self, outputs, targets):
        """
        Inputs:
            outputs: dict containing the output of the model,
                the expected entries are:
                    - sem_seg: inlier semantic segmentation logits
                    - ood_score: logits of the OoD GMM Head
                    - contrast_logits: logits for the contrastive loss
                    - contrast_targets: contrastive loss targets
            targets: ground truth semantic segmentation labels
                containing OoD pixels labeles as hparams.DATA.OOD_LABEL
        """

        if self.hparams.MODEL.LOSS.NAME == "likelihood_ratio":
            return self.likelihood_ratio_loss(outputs, targets)
        else:
            raise NotImplementedError(
                f"Given loss not supported: {self.hparams.MODEL.LOSS.NAME}"
            )

    def training_step(self, batch, batch_idx):

        x, y = batch

        ood_mask = (y == self.hparams.DATA.OOD_LABEL).long()
        ood_mask[y == self.hparams.MODEL.IGNORE_INDEX] = self.hparams.MODEL.IGNORE_INDEX

        outputs = self(x, gt_ood=ood_mask)

        B, H, W = y.shape
        if outputs.sem_seg.size(2) != H or outputs.sem_seg.size(3) != W:
            outputs.sem_seg = F.interpolate(
                outputs.sem_seg, size=(H, W), mode="bilinear", align_corners=False
            )
            outputs.ood_score = F.interpolate(
                outputs.ood_score, size=(H, W), mode="bilinear", align_corners=False
            )

        loss = self.loss_function(outputs, ood_mask)

        self.log("train/loss", loss.item(), on_epoch=True, on_step=True)

        # NOTE: this is used to monitor GMM components learning only
        if self.hparams.MODEL.OOD_HEAD.NAME == "GMMSegHead":
            # extract the norms of the gaussian components of ood_head and log them
            # (num_classes, num_components, embedding_dim)
            means = self.ood_head.means

            # compute the average pairwise distance between num_components
            # of each class
            num_classes, num_components, embedding_dim = means.size()
            means = means.view(num_classes * num_components, embedding_dim)
            means = F.normalize(means, p=2, dim=1)
            pairwise_distances = torch.cdist(means, means, p=2)
            # log the mean pairwise distance
            self.log(
                "train/mean_pairwise_distance",
                pairwise_distances.mean(),
                on_epoch=True,
                on_step=True,
            )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        NOTE: for now evaluation is either OoD or inlier
        """
        x, y = batch

        output = self.sliding_window_inference(
            x,
            y.shape,
            num_classes=self.segmentor.hparams.MODEL.NUM_CLASSES,
            window_size=self.hparams.SOLVER.EVAL_WINDOW_SIZE,
            stride=self.hparams.SOLVER.EVAL_STRIDE,
            return_ood=self.hparams.DATA.EVAL_DATASET != "cityscapes",
        )

        if self.hparams.DATA.EVAL_DATASET == "cityscapes":

            B, H, W = y.shape
            if output.sem_seg.size(2) != H or output.sem_seg.size(3) != W:
                output.sem_seg = F.interpolate(
                    output.sem_seg, size=(H, W), mode="bilinear", align_corners=False
                )

            self.iou_metric(output.sem_seg, y)

            self.log("val_iou", self.iou_metric, on_epoch=True, on_step=True)

        else:

            ood_score = output.ood_score[:, 1]
            idd_score_under_ood = output.ood_score[:, 0]

            idd_score = output.sem_seg.max(dim=1)[0]
            lr_score = ood_score - idd_score - idd_score_under_ood

            if self.hparams.DATA.EVAL_DATASET == "ra_and_fslaf":
                self.ood_metrics_idd[dataloader_idx].update(-idd_score, y)
                self.ood_metrics_ood[dataloader_idx].update(ood_score, y)
                self.ood_metrics_llr[dataloader_idx].update(lr_score, y)
            else:
                self.ood_metrics_idd.update(-idd_score, y)
                self.ood_metrics_ood.update(ood_score, y)
                self.ood_metrics_llr.update(lr_score, y)

    def on_validation_epoch_end(self) -> None:

        if self.hparams.DATA.EVAL_DATASET != "cityscapes":
            for metric in [
                self.ood_metrics_idd,
                self.ood_metrics_ood,
                self.ood_metrics_llr,
            ]:
                if self.hparams.DATA.EVAL_DATASET == "ra_and_fslaf":
                    for idx in range(len(metric)):
                        self.log_dict(metric[idx].compute(), sync_dist=True)
                        metric[idx].reset()
                else:
                    self.log_dict(metric.compute(), sync_dist=True)
                    metric.reset()

    def test_step(self, batch, batch_idx):

        x, y = batch

        output = self.sliding_window_inference(
            x,
            y.shape,
            num_classes=self.segmentor.hparams.MODEL.NUM_CLASSES,
            window_size=self.hparams.SOLVER.EVAL_WINDOW_SIZE,
            stride=self.hparams.SOLVER.EVAL_STRIDE,
            return_ood=True,
        )

        ood_score = output.ood_score[:, 1]
        idd_score_under_ood = output.ood_score[:, 0]

        idd_score = output.sem_seg.max(dim=1)[0]
        lr_score = ood_score - idd_score - idd_score_under_ood

        self.ood_metrics_idd.update(-idd_score, y)
        self.ood_metrics_ood.update(ood_score, y)
        self.ood_metrics_llr.update(lr_score, y)

    def on_test_epoch_end(self) -> None:

        for metric in [
            self.ood_metrics_idd,
            self.ood_metrics_ood,
            self.ood_metrics_llr,
        ]:
            self.log_dict(metric.compute(), sync_dist=True)
            metric.reset()

    def sliding_window_inference(
        self,
        x,
        y_shape,
        num_classes,
        window_size,
        stride,
        return_seg=True,
        return_ood=True,
    ):
        """
        params:
        x: input image of shape (B, 3, H, W)
        y_shape: shape of the semantic segmentation label, typically a tuple containing [B, W, H]
        window_size: a pair of integers representing the size of the sliding window
        stride: a pair of integers representing the stride of the sliding window
        """

        B, H, W = y_shape
        output = edict()
        if return_seg:
            output["sem_seg"] = torch.zeros(
                (B, num_classes, H, W), device=x.device
            ).float()
        if return_ood:
            output["ood_score"] = torch.zeros(
                (B, self.hparams.MODEL.OOD_HEAD.NUM_CLASSES, H, W), device=x.device
            ).float()

        window_h, window_w = window_size
        stride_h, stride_w = stride

        h_loop = list(
            zip(
                range(0, H - window_h, stride_h),
                range(window_h, H, stride_h),
            )
        )
        w_loop = list(
            zip(
                range(0, W - window_w, stride_w),
                range(window_w, W, stride_w),
            )
        )
        # if the window and stride setup does not cover the entire image, add windows
        # that cover the remaining parts of the image
        if (H - window_h) % stride_h != 0:
            h_loop.append((H - window_h, H))
        if (W - window_w) % stride_w != 0:
            w_loop.append((W - window_w, W))

        counter = torch.zeros(y_shape, device=x.device).float()
        for i, i_end in h_loop:
            for j, j_end in w_loop:
                x_window = x[:, :, i:i_end, j:j_end]
                output_window = self(
                    x_window, return_seg=return_seg, return_ood=return_ood
                )
                # check if any of the outputs is nan
                if return_seg:
                    output["sem_seg"][:, :, i:i_end, j:j_end] += F.interpolate(
                        output_window.sem_seg,
                        size=(window_h, window_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                if return_ood:
                    output["ood_score"][:, :, i:i_end, j:j_end] += F.interpolate(
                        output_window.ood_score,
                        size=(window_h, window_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                counter[:, i:i_end, j:j_end] += 1

        if return_seg:
            output.sem_seg /= counter.unsqueeze(1)
        if return_ood:
            output.ood_score /= counter.unsqueeze(1)

        return output
