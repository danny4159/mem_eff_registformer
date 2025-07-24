import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from torchvision.models import vgg19
from packaging import version

# 기존 MINDLoss import (이미 있음)
# try:
#     from src.losses.mind_loss import MINDLoss
#     MIND_AVAILABLE = True
# except ImportError:
#     MIND_AVAILABLE = False
#     print("Warning: MINDLoss를 import할 수 없습니다.")

class GANLoss(nn.Module):
    """GAN Loss 구현 (기존 코드 기반)"""
    
    def __init__(self, gan_type='lsgan', real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0, reduction='mean'):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss(reduction=reduction)
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """WGAN loss"""
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """WGAN loss with softplus"""
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label"""
        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """Forward pass"""
        target_label = self.get_target_label(input, target_is_real)
        
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class VGG_Model(nn.Module):
    """VGG Model for feature extraction (기존 코드 기반)"""
    
    # VGG 19 layer mapping
    vgg_layer = {
        "conv_1_1": 0, "conv_1_2": 2, "pool_1": 4,
        "conv_2_1": 5, "conv_2_2": 7, "pool_2": 9,
        "conv_3_1": 10, "conv_3_2": 12, "conv_3_3": 14, "conv_3_4": 16, "pool_3": 18,
        "conv_4_1": 19, "conv_4_2": 21, "conv_4_3": 23, "conv_4_4": 25, "pool_4": 27,
        "conv_5_1": 28, "conv_5_2": 30, "conv_5_3": 32, "conv_5_4": 34, "pool_5": 36,
    }
    
    vgg_layer_inv = {
        0: "conv_1_1", 2: "conv_1_2", 4: "pool_1",
        5: "conv_2_1", 7: "conv_2_2", 9: "pool_2",
        10: "conv_3_1", 12: "conv_3_2", 14: "conv_3_3", 16: "conv_3_4", 18: "pool_3",
        19: "conv_4_1", 21: "conv_4_2", 23: "conv_4_3", 25: "conv_4_4", 27: "pool_4",
        28: "conv_5_1", 30: "conv_5_2", 32: "conv_5_3", 34: "conv_5_4", 36: "pool_5",
    }
    
    def __init__(self, listen_list=None):
        super(VGG_Model, self).__init__()
        from torchvision.models import VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.vgg_model = vgg.features
        vgg_dict = vgg.state_dict()
        vgg_f_dict = self.vgg_model.state_dict()
        vgg_dict = {k: v for k, v in vgg_dict.items() if k in vgg_f_dict}
        vgg_f_dict.update(vgg_dict)
        
        # no grad
        for p in self.vgg_model.parameters():
            p.requires_grad = False
            
        if listen_list is None or listen_list == []:
            self.listen = []
        else:
            self.listen = set()
            for layer in listen_list:
                if layer in self.vgg_layer:
                    self.listen.add(self.vgg_layer[layer])
        self.features = OrderedDict()

    def forward(self, x):
        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in self.listen:
                self.features[self.vgg_layer_inv[index]] = x
        return self.features


class Distance_Type:
    """Distance types for contextual loss"""
    L2_Distance = 0
    L1_Distance = 1
    Cosine_Distance = 2


class Contextual_Loss(nn.Module):
    """Contextual Loss (Style Loss) - 기존 코드 기반"""
    
    def __init__(
        self,
        layers_weights,
        vgg=True,
        crop_quarter=True,
        max_1d_size=10000,
        distance_type=Distance_Type.Cosine_Distance,
        b=1.0,
        h=0.5,
    ):
        super(Contextual_Loss, self).__init__()
        listen_list = []
        self.layers_weights = {}
        try:
            listen_list = layers_weights.keys()
            self.layers_weights = layers_weights
        except:
            pass
            
        if vgg:
            self.vgg_pred = VGG_Model(listen_list=listen_list)
        else:
            raise NotImplementedError("Only VGG is supported for now")
            
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.h = h

    def forward(self, images, gt):
        if images.shape[1] == 1 and gt.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)

        assert (
            images.shape[1] == 3 and gt.shape[1] == 3
        ), "VGG model takes 3 channel images."

        if images.device.type == "cpu":
            loss = torch.zeros(1)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone() for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
        else:
            id_cuda = torch.cuda.current_device()
            loss = torch.zeros(1).cuda(id_cuda)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone().cuda(id_cuda) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
            vgg_gt = {k: v.cuda(id_cuda) for k, v in vgg_gt.items()}

        for key in self.layers_weights.keys():
            N, C, H, W = vgg_images[key].size()

            if self.crop_quarter:
                vgg_images[key] = self._crop_quarters(vgg_images[key])
                vgg_gt[key] = self._crop_quarters(vgg_gt[key])

            if H * W > self.max_1d_size**2:
                vgg_images[key] = self._random_pooling(
                    vgg_images[key], output_1d_size=self.max_1d_size
                )
                vgg_gt[key] = self._random_pooling(
                    vgg_gt[key], output_1d_size=self.max_1d_size
                )
                
            loss_t = self.calculate_CX_Loss(vgg_images[key], vgg_gt[key])
            loss += loss_t * self.layers_weights[key]
            
        return loss

    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = Contextual_Loss._move_to_current_device(indices)
        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _move_to_current_device(tensor):
        if tensor.device.type == "cuda":
            id = torch.cuda.current_device()
            return tensor.cuda(id)
        return tensor

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        N, C, H, W = feats.size()
        feats_sample, indices = Contextual_Loss._random_sampling(
            feats, output_1d_size**2, None
        )
        res = feats_sample.view(N, C, output_1d_size, output_1d_size)
        return res

    @staticmethod
    def _crop_quarters(feature):
        N, fC, fH, fW = feature.size()
        quarters_list = []
        quarters_list.append(feature[..., 0 : round(fH / 2), 0 : round(fW / 2)])
        quarters_list.append(feature[..., 0 : round(fH / 2), round(fW / 2) :])
        quarters_list.append(feature[..., round(fH / 2) :, 0 : round(fW / 2)])
        quarters_list.append(feature[..., round(fH / 2) :, round(fW / 2) :])
        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _centered_by_T(I, T):
        mean_T = (
            T.mean(dim=0, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        return I - mean_T, T - mean_T

    @staticmethod
    def _normalized_L2_channelwise(tensor):
        norms = tensor.norm(p=2, dim=1, keepdim=True)
        return tensor / norms

    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        I_features, T_features = Contextual_Loss._centered_by_T(I_features, T_features)
        I_features = Contextual_Loss._normalized_L2_channelwise(I_features)
        T_features = Contextual_Loss._normalized_L2_channelwise(T_features)

        N, C, H, W = I_features.size()
        cosine_dist = []
        for i in range(N):
            T_features_i = (
                T_features[i].view(1, 1, C, H * W).permute(3, 2, 0, 1).contiguous()
            )
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            cosine_dist.append(dist)
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)
        return cosine_dist

    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """Normalizing the distances"""
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon)
        return relative_dist

    def calculate_CX_Loss(self, I_features, T_features, average_over_scales=True, weight=None):
        I_features = Contextual_Loss._move_to_current_device(I_features)
        T_features = Contextual_Loss._move_to_current_device(T_features)

        raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - relative_distance) / self.h)
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)
        
        if average_over_scales:
            max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]
            CS = torch.mean(max_gt_sim, dim=1)
            if weight is not None:
                CX_loss = torch.sum(-weight * torch.log(CS))
            else:
                CX_loss = torch.mean(-torch.log(CS))
            return CX_loss
        else:
            max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]
            CS = torch.mean(max_gt_sim, dim=1)
            CX_loss = -torch.log(CS)
            return CX_loss


class PatchNCELoss(nn.Module):
    """PatchNCE Loss - 기존 코드 기반"""
    
    def __init__(self, nce_includes_all_negatives_from_minibatch, nce_T, batch_size):
        super().__init__()
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


# Implementation of MINDLoss for SC-GAN from https://github.com/HeranYang/sc-cyclegan/tree/master
# Unpaired brain MR-to-CT synthesis using a structure-constrained CycleGAN, Yang et al., 2018 MICCAI
# 1. SC-GAN : GAN_loss + lam1 * Cycle_loss + lam2 * MIND_L1 Loss
# MIND : sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7, lam2=5

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_gausian_filter(sigma, sz):
    xpos, ypos = torch.meshgrid(torch.arange(sz), torch.arange(sz))
    output = torch.ones([sz, sz, 1, 1])
    midpos = sz // 2
    d = (xpos - midpos) ** 2 + (ypos - midpos) ** 2
    gauss = torch.exp(-d / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return gauss


def gaussian_filter(img, n, sigma):
    """
    img: image tensor of size (1, 1, height, width)
    n: size of the Gaussian filter (n, n)
    sigma: standard deviation of the Gaussian distribution
    """
    # Create a Gaussian filter
    gaussian_filter = get_gausian_filter(sigma, n)
    # Add extra dimensions for the color channels and batch size
    gaussian_filter = gaussian_filter.view(1, 1, n, n)
    gaussian_filter = gaussian_filter.to(img.device)
    # Perform 2D convolution
    filtered_img = F.conv2d(img, gaussian_filter, padding=n // 2)
    return filtered_img


def Dp(image, sigma, patch_size, xshift, yshift):
    shift_image = torch.roll(image, shifts=(xshift, yshift), dims=(-1, -2))
    diff = image - shift_image
    diff_square = diff**2
    res = gaussian_filter(diff_square, patch_size, sigma)
    return res


def mind(image, sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7):
    reduce_size = (patch_size + neigh_size - 2) // 2
    # estimate the local variance of each pixel within the input image.
    Vimg = (
        Dp(image, sigma, patch_size, -1, 0)
        + Dp(image, sigma, patch_size, 1, 0)
        + Dp(image, sigma, patch_size, 0, -1)
        + Dp(image, sigma, patch_size, 0, 1)
    )
    Vimg = Vimg / 4 + eps * torch.ones_like(Vimg)

    # estimate the (R*R)-length MIND feature by shifting the input image by R*R times.
    xshift_vec = np.arange(-neigh_size // 2, neigh_size - neigh_size // 2)
    yshift_vec = np.arange(-neigh_size // 2, neigh_size - neigh_size // 2)

    # print(xshift_vec, yshift_vec)

    iter_pos = 0
    for xshift in xshift_vec:
        for yshift in yshift_vec:
            if (xshift, yshift) == (0, 0):
                continue
            MIND_tmp = torch.exp(
                -Dp(image, sigma, patch_size, xshift, yshift) / Vimg
            )  # MIND_tmp : 1x1x256x256
            tmp = MIND_tmp[
                ..., reduce_size:-reduce_size, reduce_size:-reduce_size, None
            ]  # 1x1x250x250x1
            output = tmp if iter_pos == 0 else torch.cat((output, tmp), -1)
            iter_pos += 1

    # normalization.
    output = torch.divide(output, torch.max(output, dim=-1, keepdim=True)[0])

    return output


class MINDLoss(nn.Module):
    def __init__(self, sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7):
        super(MINDLoss, self).__init__()
        self.sigma = sigma
        self.eps = eps
        self.neigh_size = neigh_size
        self.patch_size = patch_size

    def forward(self, pred, gt):
        pred_mind = mind(pred, self.sigma, self.eps, self.neigh_size, self.patch_size)
        gt_mind = mind(gt, self.sigma, self.eps, self.neigh_size, self.patch_size)
        mind_loss = F.l1_loss(pred_mind, gt_mind)
        return mind_loss


# NOTE: PadainSynthesisLoss class is not used in the current implementation
# The individual loss functions are used directly in trainer.py instead 