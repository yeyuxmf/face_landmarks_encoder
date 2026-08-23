# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from net.realnvp import RealNVP
from config import config as cfg
class RLELoss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_dis='laplace',
                #  enable_3d_rle=False,
                 ):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_dis = q_dis
        self.flow_model = RealNVP(in_channels=2)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D*2]): Output regression,
                    including coords and sigmas.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        pred = output[:, :, :2]
        sigma = output[:, :, 2:4].sigmoid()

        error = (pred - target) / (sigma + 1e-9)
        # (B, K, 2)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1],
                                             2)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_dis in ['laplace', 'gaussian', 'strict']
            if self.q_dis == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        # if self.enable_3d_rle:
        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss = loss.mean()

        return loss.mean()
def rcal_loss(outputs_seg, hotmap):
    outputs_seg = torch.sigmoid(outputs_seg)
    loss_ml_ = mse_loss_(outputs_seg, hotmap)
    loss_dl_ = dice_loss(outputs_seg, hotmap)
    # loss_re_ = region_loss(outputs_seg, hotmap)
    loss_fl_ = loss_dl_#focal_loss(outputs_seg, hotmap)
    loss_re_ = loss_dl_

    return loss_ml_, loss_dl_, loss_re_, loss_fl_
def mse_loss_(score, target):
    mask = torch.gt(target, 0.1).float()
    pos_loss = F.mse_loss(score * mask, target * mask, reduction="sum") / torch.sum(mask)
    neg_loss = F.mse_loss(score * (1 - mask), target * (1 - mask), reduction="sum") / torch.sum(1 - mask)

    loss = pos_loss + 2 * neg_loss
    return loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def region_loss(score, target, th=0.1):
    score_ = torch.gt(score, th).float()
    target_ = torch.gt(target, th).float()

    loss = -score_ * torch.log(torch.clip(target_, 1.0 * 1e-9)) - (1 - score_) * torch.log(
        torch.clip(1 - target_, 1.0 * 1e-9))

    return loss.mean()


def focal_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    aa = torch.sum(pos_inds)

    pos_loss = (torch.log(torch.clip(pred, 1.0 * 1e-5)) * torch.pow(1 - pred, 2) * pos_inds).sum() / 3.0
    neg_loss = (torch.log(torch.clip(1 - pred, 1.0 * 1e-5))* torch.pow(pred, 2) * neg_weights * neg_inds).mean() *19

    num_pos = pos_inds.float().sum()
    # pos_loss = pos_loss.sum() / 3.0
    # neg_loss = neg_loss.mean() * 19

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss)  # / num_pos
    return loss

def edge_loss(pred, gt):

    relation = cfg.Relation -1
    pred1 = pred[:, relation[:,0], :]
    pred2 = pred[:, relation[:,1], :]

    gt1 = gt[:, relation[:,0], :]
    gt2 = gt[:, relation[:,1], :]

    pred_edge = pred1 - pred2
    gt_edge = gt1 - gt2

    loss = F.l1_loss(pred_edge, gt_edge)

    return loss *1e-3