import logging

import torch
import torch.nn.functional as F
from torch import nn
from .matcher import HungarianMatcher
from easydict import EasyDict as edict

import torch.distributed as dist
import torchvision


from torchvision import transforms



def _max_by_axis(the_list):

    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def cat(tensors, dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_uncertain_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(
                    num_boxes, num_random_points, 2, device=coarse_logits.device
                ),
            ],
            dim=1,
        )
    return point_coords


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_batch_avg(logits, label_ood):
    N, _, H, W = logits.shape
    m = logits.mean(1).mean()
    ma = -m.view(1, 1, 1).repeat(N, H, W) * label_ood
    return ma


class Mask2FormerLoss(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    ):
        """Create the criterion.
        Parameters:
        - num_classes:  number of object categories, omitting the special no-object category
        - weight_dict:  dict containing as key the names of the losses and as values their relative weight.
        - eos_coef:     relative classification weight applied to the no-object category
        - losses:       list of all the losses to be applied. See get_loss for list of available losses.
        - [num_points, oversample_ratio, importance_sample_ratio]: PointRend Params
        - smoothness_score: score for which to apply local smoothness and sparsity losses
        - outlier_loss_target: the score function used in outlier loss with explicit outlier supervision
        - inlier_upper_threshold: the value below which the inlier pixels' target score should be pushed
                                , that is, the loss is higher if inlier score goes above this threshold
        - outlier_lower_threshold: the value above which the outlier pixels' target score should be pushed
                                , that is, the loss is higher if outlier score goes below this threshold

        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(
            cost_class=2.0,  # class_weight
            cost_mask=5.0,  # mask_weight
            cost_dice=5.0,  # dice_weight,
            num_points=12544,  # cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def smoothness_loss(self, outputs, targets, indices, num_masks):
        """
        Smoothness regularization loss to encourage nearby pixels to have similar negative
        logit sum.

        Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/losses.py

        TODO: in case of extra memory consumption, add POINT REND filtering
        """
        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_logits = mask_logits.sigmoid()
        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.smoothness_score == "nls":
            score = -logits.sum(dim=1)  # -> (B, H, W)
        elif self.smoothness_score == "energy":
            score = -torch.logsumexp(logits, dim=1)  # -> (B, H, W)
        elif self.outlier_loss_target == "softmax_entropy":
            score = torch.special.entr(logits.softmax(dim=1)).sum(dim=1)
        else:
            raise ValueError(f"Undefined Smoothness Score: f{self.smoothness_score}")

        score_h_shifted = torch.zeros_like(score)
        score_h_shifted[:, :-1, :] = score[:, 1:, :]
        score_h_shifted[:, -1, :] = score[:, -1, :]

        score_w_shifted = torch.zeros_like(score)
        score_w_shifted[:, :, :-1] = score[:, :, 1:]
        score_w_shifted[:, :, -1] = score[:, :, -1]

        loss = (
            torch.sum((score_h_shifted - score) ** 2)
            + torch.sum((score_w_shifted - score) ** 2)
        ) / 2

        return {"smoothness_loss": loss}

    def sparsity_loss(self, outputs, targets, indices, num_masks):
        """
        Used to encourage sparsity of the negative logits sum score.

        In the case of OOD supervision, targets will contain an entry
        with key: "outlier_masks". In this case, sparsity will be computed
        only for OOD regions, following the same method as the source below.

        Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/losses.py

        TODO: in case of extra memory consumption, add POINT REND filtering
        """
        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_logits = mask_logits.sigmoid()
        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.smoothness_score == "nls":
            score = -logits.sum(dim=1)  # -> (B, H, W)
        elif self.smoothness_score == "energy":
            score = -torch.logsumexp(logits, dim=1)  # -> (B, H, W)
        elif self.outlier_loss_target == "softmax_entropy":
            score = torch.special.entr(logits.softmax(dim=1)).sum(dim=1)
        else:
            raise ValueError(f"Undefined Smoothness Score: f{self.smoothness_loss}")

        if "outlier_masks" in targets[0]:
            outlier_masks = torch.cat(
                [x["outlier_masks"].unsqueeze(0) for x in targets], dim=0
            )  # -> (B, H, W)
            ood_mask = outlier_masks == 1
            score = F.interpolate(
                score.unsqueeze(1),
                size=outlier_masks.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)
            loss = torch.mean(torch.norm(score[ood_mask], dim=0))
        else:
            loss = torch.tensor(0.0, device=score.device)

        return {"sparsity_loss": loss}

    def outlier_loss(self, outputs, targets, indices, num_masks):
        """
        This loss is used with outlier supervision in order to explicitly minimize anomaly
        score for inlier pixels and maximize it for outlier pixels

        Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/losses.py
        """

        outlier_masks = torch.cat(
            [x["outlier_masks"].unsqueeze(0) for x in targets], dim=0
        )  # -> (B, H, W)

        ood_mask = outlier_masks == 1
        id_mask = outlier_masks == 0

        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_logits = mask_logits.sigmoid()

        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.outlier_loss_target == "nls":
            if self.score_norm == "sigmoid":
                score = logits.sigmoid()
            elif self.score_norm == "tanh":
                score = logits.tanh()
            else:
                score = logits
            score = -score.sum(dim=1)  # -> (B, H, W)
        elif self.outlier_loss_target == "energy":
            score = -torch.logsumexp(logits, dim=1)  # -> (B, H, W)
        elif self.outlier_loss_target == "softmax_entropy":
            score = torch.special.entr(logits.softmax(dim=1)).sum(dim=1)
        elif self.outlier_loss_target == "sum_entropy":
            score = torch.special.entr(
                logits.div(logits.sum(dim=1, keepdims=True))
            ).sum(dim=1)
        else:
            raise ValueError(
                f"Undefined Outlier Target Score: f{self.outlier_loss_target}"
            )

        score = F.interpolate(
            score.unsqueeze(1),
            size=outlier_masks.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

        ood_score = score[ood_mask]
        id_score = score[id_mask]

        if self.outlier_loss_func == "squared_hinge":
            loss = torch.pow(F.relu(id_score - self.inlier_upper_threshold), 2).mean()
            if ood_mask.sum() > 0:
                loss = (
                    loss
                    + torch.pow(
                        F.relu(self.outlier_lower_threshold - ood_score), 2
                    ).mean()
                )
                loss = 0.5 * loss
        elif self.outlier_loss_func == "binary_cross_entropy":
            loss = 0.5 * F.binary_cross_entropy_with_logits(
                score, ood_mask.float()
            )  # NOTE: try score + 1 to make it 1-\sigma(others)

        elif self.outlier_loss_func == "mse":
            id_up_thr_vec = torch.tensor(self.inlier_upper_threshold).to(
                id_score.device
            )
            id_up_thr_vec = id_up_thr_vec.repeat(id_score.shape)

            loss = F.mse_loss(id_score, id_up_thr_vec)

            if ood_mask.sum() > 0:
                ood_low_thr_vec = torch.tensor(self.outlier_lower_threshold).to(
                    ood_score.device
                )
                ood_low_thr_vec = ood_low_thr_vec.repeat(ood_score.shape)

                loss = loss + F.mse_loss(ood_score, ood_low_thr_vec)
                loss = 0.5 * loss

        elif self.outlier_loss_func == "l1":
            id_up_thr_vec = torch.tensor(self.inlier_upper_threshold).to(
                id_score.device
            )
            id_up_thr_vec = id_up_thr_vec.repeat(id_score.shape)

            loss = F.l1_loss(id_score, id_up_thr_vec)

            if ood_mask.sum() > 0:
                ood_low_thr_vec = torch.tensor(self.outlier_lower_threshold).to(
                    ood_score.device
                )
                ood_low_thr_vec = ood_low_thr_vec.repeat(ood_score.shape)

                loss = loss + F.l1_loss(ood_score, ood_low_thr_vec)
                loss = 0.5 * loss

        elif self.outlier_loss_func == "kl":
            score = F.interpolate(
                logits,
                size=outlier_masks.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            K = logits.shape[1]

            score = score.permute(0, 2, 3, 1).reshape(-1, K)

            id_mask = id_mask.view(-1).unsqueeze(-1)
            ood_mask = ood_mask.view(-1).unsqueeze(-1)

            id_score = torch.mul(score, id_mask)
            ood_score = torch.mul(score, ood_mask)

            sorted_id = id_score.sort(dim=-1, descending=True)[0].log_softmax(dim=-1)
            sorted_id = sorted_id.log_softmax(dim=-1)
            target_id = torch.zeros_like(sorted_id)
            target_id[:, 0] = 1.0
            target_id = target_id.softmax(dim=-1)

            loss = F.kl_div(sorted_id, target_id)

            if ood_mask.sum() > 0:

                sorted_ood = ood_score.sort(dim=-1, descending=True)[0].log_softmax(
                    dim=-1
                )
                target_ood = torch.zeros_like(sorted_ood).softmax(dim=-1)

                loss = loss + F.kl_div(sorted_ood, target_ood)
                loss = 0.5 * loss

        else:
            raise ValueError(
                f"Undefined Outlier Loss Function: f{self.outlier_loss_func}"
            )

        return {"outlier_loss": loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "smoothness": self.smoothness_loss,
            "sparsity": self.sparsity_loss,
            "outlier": self.outlier_loss
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def maskformify_targets(self, targets):
        """
        Transform the targets to the mask2former format

        input targets are of shape (B, H, W),

        return targets in the form: (B, edict(labels=[C], masks=[C, H, W])
        """
        B, H, W = targets.shape
        new_targets = []
        for b in range(B):
            ms = []
            classes = torch.unique(targets[b])
            classes = classes[classes != 255]
            for c in classes:
                ms.extend([targets[b] == c])
            
            if len(ms) == 0:
                masks = torch.zeros((1, H, W), device=targets.device)
            else:
                masks = torch.stack(ms)
            new_targets.append(edict(labels=classes, masks=masks))

        return new_targets

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # TODO: transform the data at this point to the mask2former format

        targets = self.maskformify_targets(targets)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # Dense Hybrid loss is computed only for the last layer
                    if loss == "densehybrid":
                        continue
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
