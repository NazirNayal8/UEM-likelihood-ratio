import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from .modules import create_model, MODELS
from torchmetrics import JaccardIndex
from .schedulers.poly import get_polynomial_decay_schedule_with_warmup
from .losses.mask2former import Mask2FormerLoss
from easydict import EasyDict as edict




class SegmentationModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        self.backbone = create_model(self.hparams.MODEL.BACKBONE, MODELS)

        self.segmentation_head = create_model(
            self.hparams.MODEL.SEGMENTATION_HEAD, MODELS
        )

        self.iou_metric = JaccardIndex(
            task="multiclass",
            num_classes=self.hparams.MODEL.NUM_CLASSES,
            ignore_index=self.hparams.MODEL.IGNORE_INDEX,
        )

        if self.hparams.MODEL.LOSS.NAME == "maskformer":

            self.criterion = create_model(self.hparams.MODEL.LOSS, edict(maskformer=Mask2FormerLoss))
            new_weight_dict = edict(self.hparams.MODEL.LOSS.WEIGHT_DICT.copy())
            for i in range(self.hparams.MODEL.BACKBONE.LEARNABLE_PARAMS.TRANSFORMER_PREDICTOR_CONFIG.DEC_LAYERS):
                for k,v in self.hparams.MODEL.LOSS.WEIGHT_DICT.items():
                    new_weight_dict.update({f"{k}_{i}": v})
            self.criterion.weight_dict = new_weight_dict 


    @classmethod
    def distributed_consistency(self, config, args):
        """
        If training is distributed, make sure the paramter is set in the OoD Head as well
        """
        if args.distributed:
            config.MODEL.SEGMENTATION_HEAD.DISTRIBUTED_TRAINING = True
        else:
            config.MODEL.SEGMENTATION_HEAD.DISTRIBUTED_TRAINING = False

        return config

    @classmethod
    def embedding_dim_consistency(self, config, args):

    # make sure embedding dim given from backbone is same as expected in
    # the segmentation head
        if config.MODEL.BACKBONE.LEARNABLE_PARAMS is not None:
            
            config.MODEL.SEGMENTATION_HEAD.EMBEDDING_DIM = (
                config.MODEL.BACKBONE.LEARNABLE_PARAMS.OUTPUT_DIM
            )
        else:
            config.MODEL.SEGMENTATION_HEAD.EMBEDDING_DIM = config.MODEL.BACKBONE.EMBED_DIM

        return config
    
    @classmethod
    def feature_names_consistency(self, config, args):
        """
        Ensure the feature names are consistent in non-DinoV2 models between
        the backbone output and the learnabler layers (decoder)
        """
        
        if config.MODEL.BACKBONE.NAME != "DINOv2":
            config.MODEL.BACKBONE.LEARNABLE_PARAMS.FEATURE_NAMES = config.MODEL.BACKBONE.OUT_FEATURES

        return config

    @classmethod
    def apply_consistency(self, config, args):

        config = self.embedding_dim_consistency(config, args)
        config = self.distributed_consistency(config, args)
        config = self.feature_names_consistency(config, args)

        return config

    def forward(self, x, gt_semantic_seg=None):
        return self.segmentation_head(self.backbone(x), gt_semantic_seg=gt_semantic_seg)

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

    def gmm_seg_loss(self, outputs, targets):

        sem_seg = outputs.sem_seg
        contrast_logits = outputs.contrast_logits
        contrast_targets = outputs.contrast_targets

        sem_seg_loss = F.cross_entropy(
            sem_seg, targets, ignore_index=self.hparams.MODEL.IGNORE_INDEX
        )
        contrast_loss = F.cross_entropy(
            contrast_logits,
            contrast_targets.long(),
            ignore_index=self.hparams.MODEL.IGNORE_INDEX,
        )

        total_loss = (
            sem_seg_loss + self.hparams.MODEL.LOSS.CONTRAST_LOSS_WEIGHT * contrast_loss
        )

        return total_loss
    
    def sem_seg_loss(self, outputs, targets):

        sem_seg = outputs.sem_seg
        sem_seg_loss = F.cross_entropy(
            sem_seg, targets, ignore_index=self.hparams.MODEL.IGNORE_INDEX
        )

        return sem_seg_loss
    
    def maskformer_loss(self, output, targets):
        
        losses = self.criterion(output, targets)
        
        total_loss = None
        for name, loss in losses.items():
            if name in self.criterion.weight_dict:
                if total_loss is None:
                    total_loss = loss * self.criterion.weight_dict[name]
                else:
                    total_loss = total_loss + loss * self.criterion.weight_dict[name]

                self.log(f"m2f_losses/{name}", loss.detach(), on_epoch=True, on_step=True)

        return total_loss


    def loss_function(self, outputs, targets):

        if self.hparams.MODEL.LOSS.NAME == "gmm_seg":
            return self.gmm_seg_loss(outputs, targets)
        elif self.hparams.MODEL.LOSS.NAME == "sem_seg":
            return self.sem_seg_loss(outputs, targets)
        elif self.hparams.MODEL.LOSS.NAME == "maskformer":
            return self.maskformer_loss(outputs, targets)
        else:
            raise NotImplementedError(
                f"Given loss function not supported: {self.hparams.MODEL.LOSS.NAME}"
            )

    def training_step(self, batch, batch_idx):

        x, y = batch

        outputs = self(x, gt_semantic_seg=y)

        B, H, W = y.shape
        if outputs.sem_seg.size(2) != H or outputs.sem_seg.size(3) != W:
            outputs.sem_seg = F.interpolate(
                outputs.sem_seg, size=(H, W), mode="bilinear", align_corners=False
            )

        loss = self.loss_function(outputs, y)

        self.log("train/loss", loss.detach(), on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        if self.hparams.SOLVER.EVAL_MODE == "sliding_window":
            output_sem_seg = self.sliding_window_inference(
                x,
                y.shape,
                window_size=self.hparams.SOLVER.EVAL_WINDOW_SIZE,
                stride=self.hparams.SOLVER.EVAL_STRIDE,
            )

        B, H, W = y.shape
        if output_sem_seg.size(2) != H or output_sem_seg.size(3) != W:
            
            output_sem_seg = F.interpolate(
                output_sem_seg, size=(H, W), mode="bilinear", align_corners=False
            )

        self.iou_metric(output_sem_seg, y)

        self.log("val_iou", self.iou_metric, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):

        x, y = batch
        B, H, W = y.shape

        output_sem_seg = self.sliding_window_inference(
            x,
            y.shape,
            window_size=self.hparams.SOLVER.EVAL_WINDOW_SIZE,
            stride=self.hparams.SOLVER.EVAL_STRIDE,
        )

        if output_sem_seg.size(2) != H or output_sem_seg.size(3) != W:
            output_sem_seg = F.interpolate(
                output_sem_seg, size=(H, W), mode="bilinear", align_corners=False
            )

        self.iou_metric(output_sem_seg, y)

        self.log("test_iou", self.iou_metric, on_epoch=True, on_step=True)

    def sliding_window_inference(self, x, y_shape, window_size, stride):
        """
        params:
        x: input image of shape (B, 3, H, W)
        y_shape: shape of the semantic segmentation label, typically a tuple containing [B, W, H]
        window_size: a pair of integers representing the size of the sliding window
        stride: a pair of integers representing the stride of the sliding window
        """
        B, H, W = y_shape
        output_sem_seg = torch.zeros(
            (B, self.hparams.MODEL.NUM_CLASSES, H, W), device=x.device
        ).float()
        window_h, window_w = window_size
        stride_h, stride_w = stride
        counter = torch.zeros(y_shape, device=x.device).float()
        h_loop = list(zip(
            range(0, H - window_h, stride_h),
            range(window_h, H, stride_h),
        ))
        w_loop = list(zip(
            range(0, W - window_w, stride_w),
            range(window_w, W, stride_w),
        ))
        # if the window and stride setup does not cover the entire image, add windows
        # that cover the remaining parts of the image
        if (H - window_h) % stride_h != 0:
            h_loop.append((H - window_h, H))
        if (W - window_w) % stride_w != 0:
            w_loop.append((W - window_w, W))
        for i, i_end in h_loop:
            for j, j_end in w_loop:
                x_window = x[:, :, i : i_end, j : j_end]
                output_sem_seg_window = self(x_window)
                output_sem_seg[
                    :, :, i : i_end, j : j_end
                ] += F.interpolate(
                    output_sem_seg_window.sem_seg,
                    size=(window_h, window_w),
                    mode="bilinear",
                    align_corners=False,
                )
                counter[:, i : i_end, j : j_end] += 1        
    
        output_sem_seg /= counter.unsqueeze(1)

        return output_sem_seg

