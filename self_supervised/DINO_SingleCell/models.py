from defaults.bases import *
from torch.cuda.amp import autocast

import math
import random
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from .transformers_utils import *

def cosine_scheduler(t, max_value, min_value, T):
    return min_value + 0.5 * (max_value - min_value) * (1 + np.cos(np.pi * t / T))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
        Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DINOHeadCells(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_cells,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256
    ):
        super().__init__()

        self.n_cells = n_cells
        self.in_dim = in_dim
        self.out_dim = out_dim

        if self.n_cells > 1:
            print('Aggregate embeddings using average.')

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        # self.aggregation_layer = AggregationTransformer()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # print("head: ", x.shape)
        assert self.in_dim == x.shape[-1]
        if self.n_cells > 1:
            B = int(x.shape[0] / self.n_cells)
            x = x.view(B, self.n_cells, self.in_dim)
            # print("head reshape: ", x.shape, "\n")
            x = torch.mean(x, dim=1)
            # x = self.aggregation_layer(x)
            # print("after affregation shape: ", x.shape)

        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLossMultiCenterCrops(nn.Module):
    def __init__(
        self,
        out_dim,
        multi_center_training,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        num_domains=None,
        examples_in_each_domain=None,
        device_id=None,
        only_cross_domain=False,
        dino_loss=True,
        barlow_loss=False,
        barlow_loss_weight=0.2,
        barlow_lambda_off_diag=1e-3,
        barlow_loss_batch_com=False,
        update_centering=True,
#         koleo_loss=True
    ):

        super().__init__()
        self.iter = 0
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum_anti = 1.0 - center_momentum
        self.ncrops = ncrops
        self.out_dim = out_dim
        self.dino_loss_scaling = torch.log(torch.tensor(self.out_dim))
        self.num_domains = num_domains
        self.examples_in_each_domain = examples_in_each_domain
        self.only_cross_domain = only_cross_domain
        self.barlow_loss = barlow_loss
        self.barlow_loss_batch_com = barlow_loss_batch_com
        self.dino_loss = dino_loss
        self.device_id = device_id
        self.multi_center_training = multi_center_training
        self.barlow_lambda_off_diag = barlow_lambda_off_diag
#         self.koleo_loss = koleo_loss

        self.update_centering = update_centering

        if (self.num_domains is not None) and self.multi_center_training:
            self.register_buffer("center", torch.zeros(self.num_domains, out_dim))
            self.domain_wise_centering = True
            self.examples_in_each_domain = (
                torch.tensor([self.examples_in_each_domain], dtype=torch.float32)
                .t()
                .to(self.device_id)
            )
        else:
            self.domain_wise_centering = False  # Changes if num_domains is not none
            self.register_buffer("center", torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, int(warmup_teacher_temp_epochs)
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

#         nwarmup = 10
        nwarmup = 30
#         steps = np.arange(0, 50)
        steps = np.arange(0, nepochs - 2 * nwarmup)
        min_value = 0
        max_value = 1
#         self.cross_domain_temp = [1 for i in range(nwarmup)] + [cosine_scheduler(t, max_value, min_value, nepochs) for t in steps]
        self.cross_domain_temp = [1 for i in range(nwarmup)] + \
            [cosine_scheduler(t, max_value, min_value, nepochs) for t in steps] + \
            [0 for i in range(nwarmup)]
#             [0 for i in range(140)]

        if self.barlow_loss:
            # defined in arguments, sets the scalar importance of the
            # barlow-loss
            self.barlow_loss_weight = barlow_loss_weight

            self.bn = nn.BatchNorm1d(
                self.out_dim, momentum=None, affine=False, track_running_stats=False
            )

            # just a scaling factor that makes sure the loss is 1 if doing
            # poorly, expected value for on diagonal is 1.0 while off diag is 1
            self.barlow_scaling_factor = (
                1.0 * (1.0 - self.barlow_lambda_off_diag)
                + 0.01 * self.barlow_lambda_off_diag
            )

#         if self.koleo_loss:
#             self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square
        # matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def sharpen_and_chunk_student_input(self, student_output):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        return student_out

    # KEEP !
    def sharpen_and_chunk_teacher_input(self, teacher_output, domain_center, temp):
#         print(teacher_output.shape, domain_center.shape)
        teacher_centered = (teacher_output - domain_center) / temp
        teacher_out = F.softmax(teacher_centered, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        teacher_centered = teacher_centered.detach().chunk(2)
        return teacher_out, teacher_centered

#     def get_domainwise_centers(self, domain_belonging):
#         domain_belonging = torch.cat(domain_belonging, 0).to(
#             self.device_id, non_blocking=True
#         )
# #         print("domain_belonging: ", domain_belonging.shape)
# #         domain_center = self.get_centers(
# #             len(domain_belonging),
# #             self.center,
# #             self.num_domains,
# #             domain_belonging,
# #             self.out_dim,
# #         )
#         domain_center = self.get_centers(
#             len(domain_belonging),
#             self.center,
#             self.num_domains,
#             domain_belonging,
#             self.out_dim,
#         )
#
# #         print("domain_center: ", domain_center.shape)
#
#         return domain_center, domain_belonging

    def get_domainwise_centers(self, domain_belonging, embeddings):
        domain_belonging = torch.cat(domain_belonging, 0).to(
                    self.device_id, non_blocking=True
                )

        # Normalize centers and embeddings for cosine similarity
        normalized_centers = F.normalize(self.center, dim=-1)
        normalized_embeddings = F.normalize(embeddings, dim=-1)

        distances = torch.cdist(normalized_embeddings, normalized_centers)

        # Compute similarity between embeddings and all centers
        closest_indices = torch.argmin(distances, dim=1)

        closest_centers = self.center[closest_indices]

        return closest_centers, domain_belonging


    def calculate_barlow_loss(
        self,
        teacher_centered,
        student_out,
        i_t,
        i_s,
        batch_size,
        distribute_before_batch_norm=False,
    ):

        if distribute_before_batch_norm:

            if ddp_is_on():
                dist_teacher_centered = dist_gather(teacher_centered[i_t], cat_dim=-1)
                synchronize()
                dist_student_out = dist_gather(student_out[i_s], cat_dim=-1)
                synchronize()
            else:
                dist_teacher_centered = teacher_centered[i_t]
                dist_student_out = student_out[i_s]

            # Do we want centring as an ablation?
            # Should we use the centered teacher output?
            c = self.bn(dist_teacher_centered).T @ self.bn(dist_student_out)

            c.div_(batch_size)

            self.iter_loss_component += 1
            if self.iter % 50 == 0 and is_rank0():

                diag = torch.diagonal(c, 0).clone().detach().cpu().numpy()
                wandb.log(
                    {
                        f"barllow_diag-{self.iter_loss_component}-rank_{torch.cuda.current_device()}": diag
                    },
                    step=self.iter,
                )

            on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
            off_diag = self.off_diagonal(c).pow_(2).mean()
            return (
                on_diag * (1.0 - self.barlow_lambda_off_diag)
                + self.barlow_lambda_off_diag * off_diag
            )

        else:

            c = self.bn(teacher_centered[i_t]).T @ self.bn(student_out[i_s])

            # sum the cross-correlation matrix between all gpus
            c.div_(batch_size)
            if ddp_is_on():
                dist.all_reduce(c)
                synchronize()

            self.iter_loss_component += 1
            if self.iter % 50 == 0 and is_rank0():

                diag = torch.diagonal(c, 0).clone().detach().cpu().numpy()
                wandb.log(
                    {
                        f"barllow_diag-{self.iter_loss_component}-rank_{torch.cuda.current_device()}": diag
                    },
                    step=self.iter,
                )

            on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
            off_diag = self.off_diagonal(c).pow_(2).mean()
            return (
                on_diag * (1.0 - self.barlow_lambda_off_diag)
                + self.barlow_lambda_off_diag * off_diag
            )

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def calculate_koleo_loss(
        self,
        student_out,
        eps=1e-8
    ):
        student_output = F.normalize(student_out, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(student_output)  # noqa: E741
        distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        return loss

    def forward(self, student_output, teacher_output, epoch, domain_belonging=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        student_out = self.sharpen_and_chunk_student_input(student_output)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        cross_domain_temp = self.cross_domain_temp[epoch]

        if self.domain_wise_centering:
            domain_center, domain_belonging = self.get_domainwise_centers(
                domain_belonging, teacher_output
            )

            teacher_out, teacher_centered = self.sharpen_and_chunk_teacher_input(
                teacher_output, domain_center, temp
            )

        else:
            domain_center = self.center

            teacher_out, teacher_centered = self.sharpen_and_chunk_teacher_input(
                teacher_output, domain_center, temp
            )

        batch_size = student_out[0].shape[0]
        batch_size = sum(dist_gather(batch_size))
        synchronize()

        barlow_loss = 0
        dino_loss = 0

        n_loss_terms = 0

        self.iter_loss_component = 0
        self.iter += 1

        for i_t, q in enumerate(teacher_out):
            for i_s in range(len(student_out)):

                if i_s == i_t:
                    # we skip cases where student and teacher operate on the
                    # same view
                    continue
#                 elif (random.random() > cross_domain_temp) and (
#                     (i_t == 0 and i_s in [2, 3, 4]) or (i_t == 1 and i_s in [5, 6, 7])
#                 ):
                elif (self.only_cross_domain) and (
                    (i_t == 0 and i_s in [2, 3, 4]) or (i_t == 1 and i_s in [5, 6, 7])
                ):
                    # if only doing cross domain learning, then skip views from
                    # the same image
                    continue

                n_loss_terms += 1
                if self.dino_loss:
                    loss = torch.sum(
                        -q * F.log_softmax(student_out[i_s], dim=-1), dim=-1
                    )
                    dino_loss += loss.mean()

                if self.barlow_loss:
                    barlow_loss += self.calculate_barlow_loss(
                        teacher_centered,
                        student_out,
                        i_t,
                        i_s,
                        batch_size,
                        distribute_before_batch_norm=self.barlow_loss_batch_com,
                    )

        if self.dino_loss:
            dino_loss /= n_loss_terms
            dino_loss /= self.dino_loss_scaling
        if self.barlow_loss:
            barlow_loss /= n_loss_terms
            barlow_loss /= self.barlow_scaling_factor

        if self.barlow_loss and self.dino_loss:
            total_loss = dino_loss * (1.0 - self.barlow_loss_weight) + barlow_loss * (
                self.barlow_loss_weight
            )  # Change back to this
        elif self.barlow_loss:
            total_loss = barlow_loss
        else:
            total_loss = dino_loss
        #
        if self.update_centering:
            if self.domain_wise_centering:
                self.update_domain_wise_centers(
                    len(domain_belonging),
                    self.center,
                    self.num_domains,
                    domain_belonging,
                    self.out_dim,
                    teacher_output,
                )
            else:
                self.update_center(teacher_output)

        if self.iter % 10 == 0 and is_rank0():
            wandb.log(
                {"barlow_loss": barlow_loss, "dino_loss": dino_loss}, step=self.iter
            )
        if self.iter % 50 == 0 and is_rank0():
            for cent in range(self.center.shape[0]):

                out = self.center[cent, :].clone().detach().cpu().numpy()
                wandb.log(
                    {
                        f"centering_vector_domain-{cent}-rank_{torch.cuda.current_device()}": out
                    },
                    step=self.iter,
                )

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = dist_average_tensor(batch_center)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

    @torch.no_grad()
    def update_domain_wise_centers(
        self,
        batch_size,
        center_values,
        num_domains,
        domain_belong,
        width,
        teacher_output,
    ):
        # Expand teacher out over the number of domains
        teac = teacher_output.expand((num_domains, -1, -1)).permute(1, 0, 2)

        # Domain center selection tensor of equal size of center tensor.
        # Basically a index tensor to select the correct centering for each
        # Mini-Batch sample

        eps = 1e-10
        # Create zero matrix of shape batch_size x num_domains
        one_hot_label = torch.zeros(batch_size, num_domains).to(
            self.device_id, non_blocking=True
        )

        # Make one hot vector of each row of the one_hot_vector_label matrix
        # based on the domain_belonging
        one_hot_label = one_hot_label.scatter(
            dim=1, index=domain_belong.unsqueeze(0).T, value=1
        )  # .type(torch.cuda.int64)

        # Expand along the width to make the matrix point wise multiplicatable
        # with the center tensor
        one_hot_label = one_hot_label.repeat(width, 1, 1).permute(1, 2, 0)

        # Sum over the mini-batch dimention to get the sum of change to apply to the centering matrix.
        # Then sum the number of instances of such domain to know how to calculate
        # the mean and how important the centering should be considered

        sum_of_teacher_outs_for_dimention = (teac * one_hot_label).sum(0)
        num_of_teacher_outs_for_dimention = (one_hot_label).sum(0)

        if ddp_is_on():
            dist.all_reduce(sum_of_teacher_outs_for_dimention)
            synchronize()
            dist.all_reduce(num_of_teacher_outs_for_dimention)
            synchronize()

        update_centers = sum_of_teacher_outs_for_dimention / (
            num_of_teacher_outs_for_dimention + eps
        )

        weight = num_of_teacher_outs_for_dimention / (
            num_of_teacher_outs_for_dimention.sum(0)
        )

        weight = weight / (
            (self.examples_in_each_domain + eps) / self.examples_in_each_domain.sum()
        )

        update_proportion = self.center_momentum + self.center_momentum_anti * (
            1 - weight
        )
        self.center = (
            self.center * update_proportion
            + update_centers * self.center_momentum_anti * weight
        )

#     def get_centers(self, batch_size, center_values, num_domains, domain_belong, width):
#
#         # Expand domain centers to third dimention covering the mini-batch size
#
#         cent = center_values.expand((batch_size, -1, -1))
#
#         # Domain center selection tensor of equal size of cent tensor. Basically a index tensor to select the correct centering for each Mini-Batch sample
#         # Create zero matrix of shape batch_size x num_domains
#         one_hot_label = torch.zeros(batch_size, num_domains).to(
#             self.device_id, non_blocking=True
#         )
#
#         # Make one hot vector of each row of the one_hot_vector_label matrix
#         # based on the domain_belonging
#         one_hot_label = one_hot_label.scatter(
#             dim=1, index=domain_belong.unsqueeze(0).T, value=1
#         )  # .type(torch.cuda.int64)
#
#         # Expand along the width to make the matrix point wise multiplicatable
#         # with the center tensor
#
#         one_hot_label = one_hot_label.repeat(width, 1, 1).permute(1, 2, 0)
#
#         # Sum over the num_domains dimension to get the corresponding center
#         # value. The sum should be over n-1 zero values and one non zero
#         # representing the image batch belongings index and correponding center
#         # value.
#
#         centers_to_use_for_domain_aligning = (cent * one_hot_label).sum(1)
#         return centers_to_use_for_domain_aligning


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the
                    # same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square
        # matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = dist_average_tensor(batch_center)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class DINOHead(nn.Module):
    # Taken from
    # https://github.com/facebookresearch/dino/blob/a52c63ba27ae15856a5d99e42c5a1b82daa902d8/vision_transformer.py#L314
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOCropsProbabilitySampling(BaseModel):
    def __init__(self, student, teacher, momentum_iters):
        super().__init__()

        self.n_classes = student.n_classes
        self.img_channels = student.img_channels
        self.num_domains = student.num_domains
        self.examples_in_each_domain = student.examples_in_each_domain
        in_dim = student.fc.in_features
        self.in_dim = in_dim

        dino_args = student.DINO if hasattr(student, "DINO") else {}
        projection_size = dino_args.get("projection_size", 2048)
        moving_average_decay = dino_args.get("moving_average_decay", 0.996)
        moving_average_decay_end = dino_args.get("moving_average_decay_end", 1.0)

        self.multi_center_training = dino_args.get("multi_center_training", True)
        self.embedding_centering = dino_args.get("embedding_centering", False)
        self.embedding_centering_lambda = dino_args.get(
            "embedding_centering_lambda", 0.5
        )

        self.update_centering_head = dino_args.get("update_centering_head", True)
        self.update_centering_features = dino_args.get(
            "update_centering_features", True
        )
#         self.koleo_weight = dino_args.get("koleo_weight", 0.05)

        print("ncrops: ", dino_args.get("ncrops"))
        self.loss_fn = DINOLossMultiCenterCrops(
            out_dim=projection_size,
            multi_center_training=self.multi_center_training,
            ncrops=dino_args.get("ncrops", 8),
            warmup_teacher_temp=dino_args.get("warmup_teacher_temp", 0.04),
            teacher_temp=dino_args.get("teacher_temp", 0.07),
            warmup_teacher_temp_epochs=dino_args.get("warmup_teacher_temp_epochs", 30),
            nepochs=dino_args.get("nepochs", 200),
            student_temp=dino_args.get("student_temp", 0.1),
            center_momentum=dino_args.get("center_momentum", 0.90),
            num_domains=self.num_domains,
            examples_in_each_domain=self.examples_in_each_domain,
            device_id=self.device_id,
            dino_loss=dino_args.get("use_dino_loss", True),
#             only_cross_domain=dino_args.get("only_cross_domain", False),
            only_cross_domain=dino_args.get("only_cross_domain", True),
            barlow_loss=dino_args.get("use_barlow_loss", True),
            barlow_loss_weight=dino_args.get("barlow_loss_weight", 0.5),
            barlow_lambda_off_diag=dino_args.get("barlow_lambda_off_diag", 7e-1),
            barlow_loss_batch_com=dino_args.get("barlow_loss_batch_com", False),
            update_centering=self.update_centering_head,
        ).cuda()

        # create online and target encoders
        self.student_encoder = student
        self.teacher_encoder = teacher
        self.student_encoder.aux_fc = None

        self.use_bn_in_head = dino_args.get("head_bn", False)
        self.norm_last_layer = dino_args.get("norm_last_layer", True)
        self.hidden_dim = dino_args.get("hidden_dim", 512)
        self.bottleneck_dim = dino_args.get("bottleneck_dim", 256)
        self.n_head_layers = dino_args.get("n_head_layers", 3)
        self.n_cells = dino_args.get("n_cells", 64)
        self.projection_size = projection_size

        print("ncells:", self.n_cells)
        # create online projectors and predictors
        self.student_encoder.fc = DINOHeadCells(
            in_dim=in_dim,
            out_dim=projection_size,
            n_cells=self.n_cells,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            nlayers=self.n_head_layers,
        )
        self.teacher_encoder.fc = DINOHeadCells(
            in_dim=in_dim,
            out_dim=projection_size,
            n_cells=self.n_cells,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            nlayers=self.n_head_layers,
        )

        self.teacher_encoder.fc.load_state_dict(
            deepcopy(self.student_encoder.fc.state_dict())
        )

#         assert moduleks_are_equal(
#             student, teacher
#         ), "The Teacher and the Student must have the same initial weights"

        # freezing teacher
        self.freeze_submodel(self.teacher_encoder)
        if self.teacher_encoder.aux_fc is not None:
            self.unfreeze_submodel(self.teacher_encoder.aux_fc)

        # init EMA
        self.ema_updater = EMA(moving_average_decay)
        self.momentum_scheduler = CosineSchedulerWithWarmup(
            base_value=moving_average_decay,
            final_value=moving_average_decay_end,
            iters=momentum_iters,
        )

        # send the BYOL wrapped model to the original model's GPU ID
        self.to(self.device_id)

    def reinit_heads(self):
        self.student_encoder.fc = DINOHeadCells(
            in_dim=self.in_dim,
            out_dim=4096,
            n_cells=self.n_cells,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            nlayers=self.n_head_layers,
        ).to(self.device_id)

        self.teacher_encoder.fc = DINOHeadCells(
            in_dim=self.in_dim,
            out_dim=4096,
            n_cells=self.n_cells,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            nlayers=self.n_head_layers,
        ).to(self.device_id)

        self.teacher_encoder.fc.load_state_dict(
            deepcopy(self.student_encoder.fc.state_dict())
        )

    def ema_update(self, it):
        self.ema_updater.beta = self.momentum_scheduler(it)
        for online_params, target_params in zip(
            self.student_encoder.backbone.parameters(),
            self.teacher_encoder.backbone.parameters(),
        ):
            target_params.data = self.ema_updater(
                online_params.detach().data, target_params.data
            )

        for online_params, target_params in zip(
            self.student_encoder.fc.parameters(), self.teacher_encoder.fc.parameters()
        ):
            target_params.data = self.ema_updater(
                online_params.detach().data, target_params.data
            )

    def update_dino_center(self, x, domain_belonging=None, epoch=0):
        with autocast(self.use_mixed_precision):
            images = [im.to(self.device_id, non_blocking=True) for im in x[:2]]
            with torch.no_grad():  # making sure that no grads are present here
                # only the 2 global views pass through the teacher
                teacher_output_backbone = self.teacher_encoder.backbone(images)
                teacher_output_backbone = teacher_output_backbone.detach()

                teacher_output_head = self.teacher_encoder.fc(teacher_output_backbone)
                teacher_output_head = teacher_output_head.detach()

                domain_center, domain_belonging = self.loss_fn.get_domainwise_centers(
                                domain_belonging
                )

                self.loss_fn.update_domain_wise_centers(
                                     batch_size=len(domain_belonging),
                                     center_values=self.loss_fn.center,
                                     num_domains=self.loss_fn.num_domains,
                                     domain_belong=domain_belonging,
                                     width=self.loss_fn.out_dim,
                                     teacher_output=teacher_output_head,
                )

            return teacher_output_backbone

    def forward(
        self,
        x,
        dmso_batch=False,
        embeddings_dmso=None,
        return_embedding=False,
        domain_belonging=None,
        labels=None,
        epoch=0,
        it=0,
        probability_calculation=False,
        return_n_last=1
    ):
        # Forward pass
        aux_outs = None
        with autocast(self.use_mixed_precision):
#             print(self.n_cells)
            if return_embedding:
                x = x.to(self.device_id, non_blocking=True)
                if return_n_last > 1:
                    teacher_out = self.teacher_encoder.backbone.forward_return_n_last_blocks(x, n=return_n_last)
                    in_dim = self.in_dim * return_n_last
                else:
                    teacher_out = self.teacher_encoder.backbone(x)
                    in_dim = self.in_dim
#                 int("teacher_out: ", teacher_out.shape)
                if self.teacher_encoder.aux_fc is not None and not probability_calculation:
#                     teacher_out = self.teacher_encoder.backbone(x)
                    print(self.n_cells)
                    if self.n_cells > 1:
#                         print("teacher_out: ", teacher_out.shape)
                        teacher_out = teacher_out.reshape((-1, self.n_cells, in_dim))
#                         print("teacher_out after reshape: ", teacher_out.shape)
                        teacher_out = torch.mean(teacher_out, dim=1)
                    if return_n_last == 1:
                        aux_outs = self.teacher_encoder.aux_fc(teacher_out)
                    else:
                        aux_outs = None
                    return None, teacher_out, aux_outs
                elif probability_calculation:
                    aux_outs = self.teacher_encoder.aux_fc(teacher_out)
                    return teacher_out, aux_outs

            images = [im.to(self.device_id, non_blocking=True) for im in x]
            with torch.no_grad():  # making sure that no grads are present here
                # only the 2 global views pass through the teacher
                teacher_output_head, teacher_output_backbone, _ = self.teacher_encoder(
                    images[:2], return_embedding=True, calc_aux_out=False
                )

                teacher_output_backbone = teacher_output_backbone.detach()
                teacher_output_head = teacher_output_head.detach()

#             print(teacher_output_backbone.shape)
#             print(teacher_output_head.shape)
            if self.teacher_encoder.aux_fc is not None:
                # taking only one view for the aux_fc (the 1st one -- no
                # solarize)
                aux_ins = (
                    teacher_output_backbone[: int(len(teacher_output_backbone) / 2)]
                    .clone()
                    .detach()
                )

                if self.n_cells > 1:
                    aux_ins = aux_ins.reshape((-1, self.n_cells, self.in_dim))
                    aux_ins = torch.mean(aux_ins, dim=1)

                aux_outs = self.teacher_encoder.aux_fc(aux_ins)

            # all views pass through the student
            student_output_head, student_output_backbone, _ = self.student_encoder(
                images, return_embedding=True, patch_drop=0.75
#                 images, return_embedding=True
            )

            if self.n_cells > 1:
                student_output_backbone = student_output_backbone.reshape((-1, self.n_cells, self.in_dim))
                student_output_backbone = torch.mean(student_output_backbone, dim=1).chunk(8)

            if self.multi_center_training:
                domain_belonging = domain_belonging
            else:
                domain_belonging = None

            loss = self.loss_fn(
                student_output_head,
                teacher_output_head,
                epoch,
                domain_belonging=domain_belonging,
            )

        koleo_loss = 0

        return loss.mean() + koleo_loss, aux_outs
