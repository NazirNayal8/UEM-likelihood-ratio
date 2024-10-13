import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from .misc import l2_normalize
from .distributions import MultivariateNormalDiag
from easydict import EasyDict as edict


@torch.no_grad()
def distributed_sinkhorn_wograd(out, sinkhorn_iterations=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t()  # K x B

    B = Q.shape[1]  # * num_pixels
    K = Q.shape[0]  # * num_components

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q = Q / sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per component must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q = Q / sum_of_rows
        Q = Q / K  # * it should be *true_distribution, and in this form, it is

        # normalize each column: total weight per sample must be 1/B
        Q = Q / torch.sum(Q, dim=0, keepdim=True)
        Q = Q / B

    Q = Q * B  # the colomns must sum to 1 so that Q is an assignment
    Q = Q.t()

    indexs = torch.argmax(Q, dim=1)
    Q = torch.nn.functional.one_hot(indexs, num_classes=Q.shape[1]).float()

    return Q, indexs


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print(
            "old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum,
                torch.norm(old_value, p=2),
                (1 - momentum),
                torch.norm(new_value, p=2),
                torch.norm(update, p=2),
            )
        )
    return update


def shifted_var(tensor, rowvar=True, bias=True):
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    # * input have already been shifted
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    # tensor: d,n
    var = (tensor**2).sum(-1)
    return factor * var


@torch.no_grad()
def concat_all_gather_wo_grad(tensor):
    """
    Concatenates all tensors from all processes on the same device as the tensor
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            xavier_init(m, distribution="uniform")


@torch.no_grad()
def rnd_sample(pop_size, num_samples, _uniform=False, _device=None):
    if _uniform:
        return torch.linspace(
            0, pop_size - 1, num_samples, dtype=torch.int64, device=_device
        )
    else:
        return torch.randperm(pop_size, dtype=torch.int64)[:num_samples]


class GMMSegHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_classes,
        num_components_per_class,
        update_interval,
        gamma_mean,
        gamma_cov,
        memory_size,
        sinkhorn_factors,  # (factor_n, factor_c, factor_p)
        max_sample_size,  # maximum number of samples in memory per component
        ignore_class=255,
        distributed_training=False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_components_per_class = num_components_per_class
        self.update_interval = update_interval
        self.gamma_mean = gamma_mean
        self.gamma_cov = gamma_cov
        # NOTE: currently this code assumed all components have the same memory size
        self.memory_size = memory_size
        self.sinkhorn_factors = sinkhorn_factors
        self.max_sample_size = max_sample_size
        self.distributed_training = distributed_training

        # memory queue
        self.register_buffer(
            "queue",
            torch.randn(
                num_classes * num_components_per_class, embedding_dim, memory_size
            ),
        )
        self.queue = F.normalize(self.queue, dim=-2)
        # the queue pointer is used to track the position of the next element to be updated in the queue
        self.register_buffer(
            "queue_ptr",
            torch.zeros(num_classes * num_components_per_class, dtype=torch.long),
        )

        self.apply(init_weights)

        # Gaussian Parameters
        self.means = nn.Parameter(
            torch.zeros(num_classes, num_components_per_class, embedding_dim),
            requires_grad=False,
        )
        trunc_normal_(self.means, std=0.02)

        self.diagonal = nn.Parameter(
            torch.ones(
                self.num_classes, self.num_components_per_class, self.embedding_dim
            ),
            requires_grad=False,
        )
        self.eye_matrix = nn.Parameter(
            torch.ones(self.embedding_dim), requires_grad=False
        )
        self.feat_norm = nn.LayerNorm(self.embedding_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.ignore_class = ignore_class

        # NOTE: should be updated after every iteration according to the reference implementation
        self.iteration_counter = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, gt_semantic_seg=None):
        """
        Inputs:
            x: (B, C, H, W) input feature map
            gt_semantic_seg: (B, H, W) ground truth segmentation map, If none, then just inference
        """
        B, C, H, W = x.size()
        # apply layer norm
        x = rearrange(x, "b c h w -> (b h w) c")
        x = self.feat_norm(x)
        x = l2_normalize(x)

        # normalize the means
        self.means.data.copy_(l2_normalize(self.means))

        log_prob = self.compute_log_prob(
            x
        )  # (B, num_classes, num_components_per_class)
        # select the max likelihood for each class from among all components
        m_prob = torch.amax(log_prob, dim=-1)  # (B, num_classes)

        # segmentation map based on the max likelihood
        out_seg = self.mask_norm(m_prob)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=B, h=H, w=W)

        # If ground truth is provided, apply update
        if gt_semantic_seg is not None:
            H_gt, W_gt = gt_semantic_seg.shape[1:]
            if H_gt != H or W_gt != W:
                gt_seg = F.interpolate(
                    gt_semantic_seg.float().unsqueeze(1), size=(H, W), mode="nearest"
                ).squeeze(1)
            else:
                gt_seg = gt_semantic_seg
            gt_seg = gt_seg.view(-1)

            contrast_logits, contrast_targets, qs = self.online_contrast(
                gt_seg, log_prob, x, out_seg
            )

            with torch.no_grad():
                # update memory
                # gather features from all GPUs
                x_mem = (
                    concat_all_gather_wo_grad(x)
                    if self.distributed_training and self.training
                    else x
                )
                gt_seg_mem = (
                    concat_all_gather_wo_grad(gt_seg)
                    if self.distributed_training and self.training
                    else gt_seg
                )
                qs_mem = (
                    concat_all_gather_wo_grad(qs)
                    if self.distributed_training and self.training
                    else qs
                )

                unique_c_list = gt_seg_mem.unique().int()
                for k in unique_c_list:
                    if k == self.ignore_class:
                        continue
                    self.dequeue_and_enqueue(
                        k.item(), x_mem, qs_mem.bool(), (gt_seg_mem == k.item())
                    )

                # update the means and covariances
                if self.iteration_counter % self.update_interval == 0:
                    self.update_gaussian_parameters(unique_c_list)

                self.iteration_counter += 1

            return edict(
                sem_seg=out_seg,
                contrast_logits=contrast_logits,
                contrast_targets=contrast_targets,
            )

        return edict(sem_seg=out_seg)

    @torch.no_grad()
    def update_gaussian_parameters(self, unique_c_list):
        """
        Inputs:
            unique_c_list: list of unique class indices
        """
        components = self.means.data.clone()
        covariances = self.diagonal.data.clone()

        for k in unique_c_list:
            if k == self.ignore_class:
                continue
            k = k if isinstance(k, int) else k.item()
            # get the indices of the k-th class
            for comp in range(self.num_components_per_class):
                q_ptr = (k * self.num_components_per_class) + comp
                q_features = self.queue[q_ptr, :, :].transpose(
                    0, 1
                )  # -> (memory_size, embedding_dim)

                f = l2_normalize(torch.sum(q_features, dim=0))  # -> (embedding_dim)

                # compute mean
                new_value = momentum_update(
                    old_value=components[k, comp, :],
                    new_value=f,
                    momentum=self.gamma_mean,
                    debug=False,
                )
                components[k, comp, :] = new_value

                # compute covariance
                shifted_features = q_features - f.unsqueeze(
                    0
                )  # -> (memory_size, embedding_dim): (input - mean)
                cov = shifted_var(shifted_features, rowvar=False)  # -> (embedding_dim)
                cov = cov + 1e-2 * self.eye_matrix
                cov = cov.sqrt()

                new_cov = momentum_update(
                    old_value=covariances[k, comp, :],
                    new_value=cov,
                    momentum=self.gamma_cov,
                    debug=False,
                )
                covariances[k, comp, :] = new_cov

            self.means = nn.Parameter(components, requires_grad=False)
            self.diagonal = nn.Parameter(covariances, requires_grad=False)

    @torch.no_grad()
    def dequeue_and_enqueue(self, k, x, q, mask):
        """
        Inputs:
            k: class index
            x: (B, C) input feature map
            q: (B, num_components_per_class) one-hot encoded membership (type is boolean)
            mask: (B) binary mask for pixels of the kth class
        """
        # dequeue and enqueue
        B, C = x.size()
        if mask is None:
            mask = torch.ones(B).detach_()

        x_k = x[mask]  # (N, C)
        q_k = q[mask]  # (N, num_components_per_class)

        for q_index in range(self.num_components_per_class):
            q_ptr = (
                k * self.num_components_per_class
            ) + q_index  # index if the q_index-th component of the k-th class
            ptr = int(self.queue_ptr[q_ptr])

            # ignore if the component q_index is not used
            if torch.sum(q_k[:, q_index]) == 0:
                continue

            assert q_k[:, q_index].shape[0] == x_k.shape[0]

            # this line below works because q_k is BOOLEAN
            x_of_q = x_k[
                q_k[:, q_index]
            ]  # (N, C): features of the q_index-th component of the k-th class

            num_samples_x_of_q = x_of_q.shape[0]
            # sanity check
            assert num_samples_x_of_q == torch.sum(q_k[:, q_index])

            # if the number of samples is greater than the max sample size, then sample uniformly
            if self.max_sample_size != -1 and num_samples_x_of_q > self.max_sample_size:
                random_sample = rnd_sample(
                    num_samples_x_of_q,
                    self.max_sample_size,
                    _uniform=True,
                    _device=x.device,
                )
                x_of_q = x_of_q[random_sample]
                num_samples_x_of_q = self.max_sample_size

            # update the queue
            if ptr + num_samples_x_of_q >= self.memory_size:
                # if updated chunk goes beyond the memory size, put whatever fits and then put the rest at the beginning
                rest_of_mem = self.memory_size - ptr
                self.queue[q_ptr, :, ptr : self.memory_size] = x_of_q[:rest_of_mem].t()
                rest_of_update = num_samples_x_of_q - rest_of_mem
                self.queue[q_ptr, :, :rest_of_update] = x_of_q[rest_of_mem:].t()
            else:
                self.queue[q_ptr, :, ptr : ptr + num_samples_x_of_q] = x_of_q.t()

            ptr = (ptr + num_samples_x_of_q) % self.memory_size
            self.queue_ptr[q_ptr] = ptr

    def online_contrast(self, gt_seg, log_prob, x, out_seg):
        """
        Inputs:
            gt_seg: (B*H*W) ground truth segmentation map that is flattened
            log_prob: (B*H*W, num_classes, num_components_per_class) log probabilities
            x: (B*H*W, C) input feature map
            out_seg: (B, num_classes, H, W) segmentation map

        Role:
            - Compute scores that measure the membership of each pixel to each component
        """
        # find pixels that are correctly classified
        pred_seg = torch.argmax(out_seg, dim=1)
        correct_mask = (pred_seg.view(-1) == gt_seg).float()

        # compute logits
        contrast_logits = log_prob.flatten(
            1
        )  # -> (B*H*W, num_classes*num_components_per_class)
        contrast_target = gt_seg.clone().float()

        # one hot encodings of memberships of each pixel to each component in their respective class
        return_qs = torch.zeros(
            size=(log_prob.size(0), self.num_components_per_class), device=gt_seg.device
        )

        # clustering for each  class
        for k in gt_seg.unique().long():
            if k == self.ignore_class:
                continue

            # get initial assignments for the k-th class
            init_q = log_prob[
                :, k, :
            ]  # -> (B*H*W, num_components_per_class): components of kth class
            init_q = init_q[
                gt_seg == k, ...
            ]  # -> (N, num_components_per_class): components of kth class for pixels of kth class, N is the number of pixels in kth class

            # the next line is added in the original code but it seems useless to me since the number of components is already fixed
            # init_q = init_q[:, :self.num_components_per_class]

            init_q = (
                init_q / torch.abs(init_q).max()
            )  # normalize the initial assignments

            # apply sinkhorn clustering
            q, indices = distributed_sinkhorn_wograd(init_q)
            # q is one-hot encoded version of index, so q is (N, num_components_per_class)
            try:
                assert torch.isnan(q).int().sum() <= 0
            except:
                # process nans
                q[torch.isnan(q)] = 0
                indices[torch.isnan(q).int().sum(dim=1) > 0] = 255 - (
                    self.num_components_per_class * k
                )

            # binary mask for pixels of the kth class
            m_k = correct_mask[gt_seg == k]  # (N)

            m_k_tile = repeat(
                m_k, "n -> n tile", tile=self.num_components_per_class
            )  # (N, num_components_per_class)

            # mask the incorrect q with zero
            q = q * m_k_tile  # -> (N, num_components_per_class)

            contrast_target[gt_seg == k] = indices.float() + (
                self.num_components_per_class * k
            )  # -> (N)

            return_qs[gt_seg == k] = q

        return contrast_logits, contrast_target, return_qs

    def compute_log_prob(self, x):
        """
        Inputs:
            x: (B, C) input feature map
        """
        covariances = self.diagonal.detach_()

        B, C = x.size()
        prob_n = []
        n_group = B // self.sinkhorn_factors[0]  # factor is 1 in config
        c_group = self.num_classes // self.sinkhorn_factors[1]  # factor is 1 in config

        # accounting for the possibility of having a group of classes
        for i in range(0, self.num_classes, c_group):
            prob_c = []
            c_means = self.means[
                i : i + c_group
            ]  # (num_classes, num_components_per_class, embedding_dim)
            c_covariances = covariances[
                i : i + c_group
            ]  # (num_classes, num_components_per_class, embedding_dim)
            c_gaussian = MultivariateNormalDiag(
                loc=c_means.view(-1, self.embedding_dim),
                scale_diag=c_covariances.view(-1, self.embedding_dim),
            )
            # accounting for the possibility of having a group of samples in the batch
            for j in range(0, B, n_group):
                log_prob = c_gaussian.log_prob(
                    x[j : j + n_group].unsqueeze(1)
                )  # (B, num_classes, num_components_per_class) NOTE check
                prob_c.append(log_prob)
            prob_c = torch.cat(prob_c, dim=0)
            prob_c = prob_c.contiguous().view(
                B, self.num_classes, self.num_components_per_class
            )
            prob_n.append(prob_c)
        probs = torch.cat(prob_n, dim=1)

        return probs.contiguous().view(
            B, self.num_classes, self.num_components_per_class
        )
