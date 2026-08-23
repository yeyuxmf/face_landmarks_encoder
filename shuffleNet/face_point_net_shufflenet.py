# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from timm.models.vision_transformer import Block
from functools import partial
from net.pos_embdb import get_2d_sincos_pos_embed, PositionalEncoding
from torch.nn.init import normal_
from net.AIFIBlock import C3k2, AIFI, C3, C2f, C2fCIB, A2C2f, Conv
from net.regression_loss import RLELoss
from config import config as cfg
from torchvision.models import shufflenet_v2_x1_0
from net.transformer import Transformer
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_proposal_pos_embed(proposals,
                           num_pos_feats=128,
                           temperature=10000):
    """Get the position embedding of proposal."""
    num_pos_feats = num_pos_feats // 2
    scale = 2 * math.pi
    dim_t = torch.arange(
        num_pos_feats, dtype=torch.float32, device=proposals.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    # N, L, 2
    proposals = proposals * scale

    # N, L, 2, 128
    pos = proposals[:, :, :, None] / dim_t
    # N, L, 2, 64, 2
    pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
    return pos


def get_cls_embed(layer2, layer3, output, coords, sizes, wcl):

    coords = coords.view(-1, 1, cfg.PointNms, 2) * 2 - 1
    coords = coords.clip(-1, 1)

    cls2 = F.grid_sample(layer2, coords, align_corners=False)
    cls3 = F.grid_sample(layer3, coords, align_corners=False)
    cls4 = F.grid_sample(output, coords, align_corners=False)


    cls2 = cls2.squeeze(dim=2).permute(0, 2, 1)
    cls3 = cls3.squeeze(dim=2).permute(0, 2, 1)
    cls4 = cls4.squeeze(dim=2).permute(0, 2, 1)


    return [None, cls2, cls3, cls4]



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class TBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        self.cv1 = Conv(c1, c1//2,1, 1)
        self.cv2 = Conv(c1//2, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Linear_with_norm(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear_with_norm, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y
class REHEADV12(nn.Module):

    def __init__(self,  c1=128, c2=256, c3=512, outc=256):
        super(REHEADV12, self).__init__()

        self.unsample1 = nn.Upsample(scale_factor=2, mode ='nearest')
        self.convck1 = A2C2f(c1=c3+c2, c2=outc, n=2 , a2=False, area=-1)
        self.unsample2 = nn.Upsample(scale_factor=2, mode ='nearest')

        self.convck2 = A2C2f(c1=c2+c1, c2=outc, n=2 , a2=False, area=-1)
        self.out1 = Conv(c1=outc, c2=outc//2, k=3, s=1)
        self.conv1 = Conv(c1=outc, c2=outc, k=3, s=2)


        self.convck3 = A2C2f(c1=c2+c2, c2=outc, n=2 , a2=False, area=-1)
        self.out2 = Conv(c1=outc, c2=outc, k=3, s=1)
        self.conv2 = Conv(c1=outc, c2=outc, k=3, s=2)

        self.convck4 =C3k2(c1=c3+c2, c2=outc, n=2 , c3k=True)
        self.out3 = Conv(c1=outc, c2=outc*2, k=3, s=1)


    def forward(self, x1, x2, x3):
        sam1 = self.unsample1(x3)
        cat1 = torch.cat([sam1, x2], dim=1)

        ck1 = self.convck1(cat1)
        sam2 = self.unsample2(ck1)
        cat2 = torch.cat([sam2, x1], dim=1)

        ck2 = self.convck2(cat2)
        out1 = self.out1(ck2)

        conv1 = self.conv1(ck2)

        cat3 = torch.cat([ck1, conv1], dim=1)

        ck3 = self.convck3(cat3)
        out2 = self.out2(ck3)

        conv2 = self.conv2(ck3)
        cat3 = torch.cat([x3, conv2], dim=1)
        ck4 = self.convck4(cat3)
        out3 = self.out3(ck4)

        return out1, out2, out3

class REHEADV11(nn.Module):

    def __init__(self,  c1=128, c2=256, c3=512, outc=256):
        super(REHEADV11, self).__init__()

        self.unsample1 = nn.Upsample(scale_factor=2, mode ='nearest')
        self.convck1 = C3k2(c1=c3+c2, c2=c2, n=2 , c3k=True)
        self.unsample2 = nn.Upsample(scale_factor=2, mode ='nearest')

        self.convck2 = C3k2(c1=c2+c1, c2=c2, n=2 , c3k=True)
        self.out1 = nn.Conv2d(in_channels=c2, out_channels=outc, kernel_size=3, padding=1)#Conv(c1=outc, c2=outc, k=3, s=1)
        self.conv1 = Conv(c1=c2, c2=c2, k=3, s=2)


        self.convck3 = C3k2(c1=c2+c2, c2=c2, n=2 , c3k=True)
        self.out2 =  nn.Conv2d(in_channels=c2, out_channels=outc, kernel_size=3, padding=1)#Conv(c1=outc, c2=outc, k=3, s=1)
        self.conv2 = Conv(c1=c2, c2=c2, k=3, s=2)

        self.convck4 =C3k2(c1=c3+c2, c2=c2, n=2 , c3k=True)
        self.out3 =  nn.Conv2d(in_channels=c2, out_channels=outc, kernel_size=3, padding=1)#Conv(c1=outc, c2=outc, k=3, s=1)


    def forward(self, x1, x2, x3):
        sam1 = self.unsample1(x3)
        cat1 = torch.cat([sam1, x2], dim=1)

        ck1 = self.convck1(cat1)
        sam2 = self.unsample2(ck1)
        cat2 = torch.cat([sam2, x1], dim=1)

        ck2 = self.convck2(cat2)
        out1 = self.out1(ck2)

        conv1 = self.conv1(ck2)

        cat3 = torch.cat([ck1, conv1], dim=1)

        ck3 = self.convck3(cat3)
        out2 = self.out2(ck3)

        conv2 = self.conv2(ck3)
        cat3 = torch.cat([x3, conv2], dim=1)
        ck4 = self.convck4(cat3)
        out3 = self.out3(ck4)

        return out1, out2, out3

class REHEADV10(nn.Module):

    def __init__(self,  c1=128, c2=256, c3=512, outc=256):
        super(REHEADV10, self).__init__()

        self.unsample1 = nn.Upsample(scale_factor=2, mode ='nearest')
        self.convck1 = C2fCIB(c1=c3+c2, c2=outc, n=3 , shortcut=True)
        self.unsample2 = nn.Upsample(scale_factor=2, mode ='nearest')

        self.convck2 = C2f(c1=c2+c1, c2=outc, n=3)
        self.out1 = Conv(c1=outc, c2=outc//2, k=3, s=1)
        self.conv1 = Conv(c1=outc, c2=outc, k=3, s=2)


        self.convck3 = C2fCIB(c1=c2+c2, c2=outc, n=3 , shortcut=True)
        self.out2 = Conv(c1=outc, c2=outc, k=3, s=1)
        self.conv2 = Conv(c1=outc, c2=outc, k=3, s=2)

        self.convck4 =C2fCIB(c1=c3+c2, c2=outc, n=3 , shortcut=True)
        self.out3 = Conv(c1=outc, c2=outc*2, k=3, s=1)


    def forward(self, x1, x2, x3):
        sam1 = self.unsample1(x3)
        cat1 = torch.cat([sam1, x2], dim=1)

        ck1 = self.convck1(cat1)
        sam2 = self.unsample2(ck1)
        cat2 = torch.cat([sam2, x1], dim=1)

        ck2 = self.convck2(cat2)
        out1 = self.out1(ck2)

        conv1 = self.conv1(ck2)

        cat3 = torch.cat([ck1, conv1], dim=1)

        ck3 = self.convck3(cat3)
        out2 = self.out2(ck3)

        conv2 = self.conv2(ck3)
        cat3 = torch.cat([x3, conv2], dim=1)
        ck4 = self.convck4(cat3)
        out3 = self.out3(ck4)

        return out1, out2, out3


class REHEADV8(nn.Module):

    def __init__(self,  c1=128, c2=256, c3=512, outc=256):
        super(REHEADV8, self).__init__()

        self.unsample1 = nn.Upsample(scale_factor=2, mode ='nearest')
        self.convck1 = C2f(c1=c3+c2, c2=outc, n=3)
        self.unsample2 = nn.Upsample(scale_factor=2, mode ='nearest')

        self.convck2 = C2f(c1=c2+c1, c2=outc, n=3)
        self.out1 = Conv(c1=outc, c2=outc//2, k=3, s=1)
        self.conv1 = Conv(c1=outc, c2=outc, k=3, s=2)


        self.convck3 = C2f(c1=c2+c2, c2=outc, n=3)
        self.out2 = Conv(c1=outc, c2=outc, k=3, s=1)
        self.conv2 = Conv(c1=outc, c2=outc, k=3, s=2)

        self.convck4 =C2f(c1=c3+c2, c2=outc, n=3)
        self.out3 = Conv(c1=outc, c2=outc*2, k=3, s=1)


    def forward(self, x1, x2, x3):
        sam1 = self.unsample1(x3)
        cat1 = torch.cat([sam1, x2], dim=1)

        ck1 = self.convck1(cat1)
        sam2 = self.unsample2(ck1)
        cat2 = torch.cat([sam2, x1], dim=1)

        ck2 = self.convck2(cat2)
        out1 = self.out1(ck2)

        conv1 = self.conv1(ck2)

        cat3 = torch.cat([ck1, conv1], dim=1)

        ck3 = self.convck3(cat3)
        out2 = self.out2(ck3)

        conv2 = self.conv2(ck3)
        cat3 = torch.cat([x3, conv2], dim=1)
        ck4 = self.convck4(cat3)
        out3 = self.out3(ck4)

        return out1, out2, out3



class REHEADV5(nn.Module):

    def __init__(self,  c1=128, c2=256, c3=512, outc=256):
        super(REHEADV5, self).__init__()


        self.conv1 = Conv(c1=c3, c2=c3, k=1, s=1)
        self.unsample1 = nn.Upsample(scale_factor=2, mode ='nearest')
        self.convck1 = C3(c1=c3+c2, c2=outc, n=3)

        self.conv2 = Conv(c1=outc, c2=outc, k=1, s=1)
        self.unsample2 = nn.Upsample(scale_factor=2, mode ='nearest')
        self.convck2 = C3(c1=c2+c1, c2=outc, n=3)

        self.out1 = Conv(c1=outc, c2=outc//2, k=3, s=1)
        self.conv3 = Conv(c1=outc, c2=outc, k=3, s=2)


        self.convck3 = C3(c1=c2+c2, c2=outc, n=3)
        self.out2 = Conv(c1=outc, c2=outc, k=3, s=1)
        self.conv4 = Conv(c1=outc, c2=outc, k=3, s=2)

        self.convck4 =C3(c1=c3+c2, c2=outc, n=3)
        self.out3 = Conv(c1=outc, c2=outc*2, k=3, s=1)


    def forward(self, x1, x2, x3):

        x3 = self.conv1(x3)
        sam1 = self.unsample1(x3)
        cat1 = torch.cat([sam1, x2], dim=1)
        ck1 = self.convck1(cat1)

        conv2 = self.conv2(ck1)
        sam2 = self.unsample2(conv2)
        cat2 = torch.cat([sam2, x1], dim=1)
        ck2 = self.convck2(cat2)

        out1 = self.out1(ck2)

        conv3 = self.conv3(ck2)

        cat3 = torch.cat([ck1, conv3], dim=1)

        ck3 = self.convck3(cat3)
        out2 = self.out2(ck3)

        conv4 = self.conv4(ck3)
        cat4 = torch.cat([x3, conv4], dim=1)
        ck4 = self.convck4(cat4)
        out3 = self.out3(ck4)

        return out1, out2, out3


class REHEADV6(nn.Module):

    def __init__(self,  c1=128, c2=256, c3=512, outc=256):
        super(REHEADV6, self).__init__()


        self.conv1 = Conv(c1=c3, c2=outc, k=1, s=1)
        self.unsample1 = nn.ConvTranspose2d(in_channels=outc,  out_channels=outc, kernel_size=2, stride=2)
        self.conv1_0 = Conv(c1=outc+c2, c2=outc, k=3, s=1)
        self.convck1 = nn.Sequential(*(Conv(c1=outc, c2=outc, k=3, s=1) for _ in range(9)))



        self.conv2 = Conv(c1=outc, c2=outc//2, k=1, s=1)
        self.unsample2 = nn.ConvTranspose2d(in_channels=outc//2,  out_channels=outc//2, kernel_size=2, stride=2)
        self.conv2_0 = Conv(c1=outc//2+c1, c2=outc//2, k=3, s=1)
        self.convck2 = nn.Sequential(*(Conv(c1=outc//2, c2=outc//2, k=3, s=1) for _ in range(9)))

        self.out1 = Conv(c1=outc//2, c2=outc//2, k=3, s=1)
        self.conv3 = Conv(c1=outc//2, c2=outc//2, k=3, s=2)
        self.conv3_0 = Conv(c1=outc+outc//2, c2=outc, k=3, s=1)
        self.convck3 = nn.Sequential(*(Conv(c1=outc, c2=outc, k=3, s=1) for _ in range(9)))


        self.out2 = Conv(c1=outc, c2=outc, k=3, s=1)

        self.conv4 = Conv(c1=outc, c2=outc, k=3, s=2)
        self.conv4_0 = Conv(c1=c3+outc, c2=outc*2, k=3, s=1)
        self.convck4 =nn.Sequential(*(Conv(c1=outc*2, c2=outc*2, k=3, s=1) for _ in range(9)))
        self.out3 = Conv(c1=outc*2, c2=outc*2, k=3, s=1)


    def forward(self, x1, x2, x3):

        conv1 = self.conv1(x3)
        sam1 = self.unsample1(conv1)
        cat1 = self.conv1_0(torch.cat([sam1, x2], dim=1))
        ck1 = self.convck1(cat1)

        conv2 = self.conv2(ck1)
        sam2 = self.unsample2(conv2)
        cat2 = self.conv2_0(torch.cat([sam2, x1], dim=1))
        ck2 = self.convck2(cat2)

        out1 = self.out1(ck2)

        conv3 = self.conv3(ck2)

        cat3 = self.conv3_0(torch.cat([ck1, conv3], dim=1))

        ck3 = self.convck3(cat3)
        out2 = self.out2(ck3)

        conv4 = self.conv4(ck3)
        cat3 = self.conv4_0(torch.cat([x3, conv4], dim=1))
        ck4 = self.convck4(cat3)
        out3 = self.out3(ck4)

        return out1, out2, out3

class get_model(nn.Module):

    def __init__(self, num_layers=None, heads=None, head_conv=None):
        super(get_model, self).__init__()
        self.scalek = 1
        self.numchannel = 256
        self.nchannel = 256#self.numchannel

        model = shufflenet_v2_x1_0(pretrained=True)

        self.backone = model

        self.aifi = AIFI(c1=464, cm=640, num_heads=8)
        self.head = REHEADV11(c1=116*self.scalek, c2=232*self.scalek, c3=464*self.scalek, outc=self.nchannel*self.scalek)

        self.out1 = nn.Conv2d(in_channels= 464*self.scalek, out_channels=68, kernel_size=1, stride=1)

        self.Tsize = torch.tensor([8, 8]).int()
        self.TsizeS = self.Tsize*2
        self.pcls_embed = nn.Parameter(torch.zeros(1, cfg.PointNms, self.nchannel), requires_grad=False)


        self.convF2 = nn.Conv2d(in_channels=128*self.scalek, out_channels=self.nchannel, kernel_size=1, stride=1, bias=False)
        self.convF3 = nn.Conv2d(in_channels=256*self.scalek, out_channels=self.nchannel, kernel_size=1, stride=1, bias=False)
        self.convF4 = nn.Conv2d(in_channels=512*self.scalek, out_channels=self.nchannel, kernel_size=1, stride=1, bias=False)

        self.bnF2 = nn.BatchNorm2d(self.nchannel, momentum=BN_MOMENTUM)
        self.bnF3 = nn.BatchNorm2d(self.nchannel, momentum=BN_MOMENTUM)
        self.bnF4 = nn.BatchNorm2d(self.nchannel, momentum=BN_MOMENTUM)

        self.linearP2 = nn.Linear(in_features=128*self.scalek, out_features=self.nchannel, bias=False)
        self.linearP3 = nn.Linear(in_features=256*self.scalek, out_features=self.nchannel, bias=False)
        self.linearP4 = nn.Linear(in_features=512*self.scalek, out_features=self.nchannel, bias=False)


        self.convP2 = nn.Conv2d(in_channels=self.numchannel // 4*self.scalek, out_channels=self.nchannel, kernel_size=1, stride=1, bias=False)
        self.convP3 = nn.Conv2d(in_channels=self.numchannel // 2*self.scalek, out_channels=self.nchannel, kernel_size=1, stride=1, bias=False)
        self.convP4 = nn.Conv2d(in_channels=self.numchannel // 1*self.scalek, out_channels=self.nchannel, kernel_size=1, stride=1, bias=False)


        self.inintwc = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]), requires_grad=True)
        self.inintwp = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]), requires_grad=True)
        self.pg = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wg = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wcl1 = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wcl2 = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)

        self.pos2 = nn.Parameter(torch.zeros(1, self.Tsize[0] * 4, self.Tsize[1] * 4,  self.numchannel // 4*self.scalek),requires_grad=False)
        self.pos3 = nn.Parameter(torch.zeros(1, self.Tsize[0] * 2, self.Tsize[1] * 2,  self.numchannel // 2*self.scalek),requires_grad=False)
        self.pos4 = nn.Parameter(torch.zeros(1, self.Tsize[0],  self.Tsize[1],  self.numchannel*self.scalek),requires_grad=False)  # fixed sin-cos embedding
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.InitEncoder = nn.ModuleList([
            Block(dim=self.nchannel, num_heads=self.nchannel // 64, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
            for i in range(2)])
        # self.InitEncoderMoe = nn.ModuleList([
        #     MoeBlock(dim=nchannel, num_heads=nchannel // 64, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(4)])

        self.linear11 = nn.Linear(self.nchannel, 256)
        self.linear12 = nn.Linear(256, 4)

        # self.DEncoder = nn.ModuleList([
        #     Block(dim=self.nchannel, num_heads=self.nchannel // 32, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(3)])
        self.DEncoder = nn.ModuleList([
            Transformer(dim=self.nchannel, num_heads=self.nchannel // 32, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
            for i in range(3)])


        # self.linear21 = nn.Linear(numchannel, 256)
        # self.linear22 = nn.Linear(256, 4)

        fc_coord_branch = []
        for _ in range(2):
            fc_coord_branch.append(nn.Linear(self.nchannel, self.nchannel))
            fc_coord_branch.append(nn.ReLU())
        fc_coord_branch.append(nn.Linear(self.nchannel, 2))
        fc_coord_branch = nn.Sequential(*fc_coord_branch)
        self.fc_coord_branches = self._get_clones(fc_coord_branch, 4)
        self.fc_coord_output_branches = self._get_clones(fc_coord_branch, 4)

        fc_sigma_branch = []
        for _ in range(2):
            fc_sigma_branch.append(nn.Linear(self.nchannel, self.nchannel))
        fc_sigma_branch.append(Linear_with_norm(self.nchannel, 2, norm=False))
        fc_sigma_branch = nn.Sequential(*fc_sigma_branch)
        self.fc_sigma_branches = self._get_clones(fc_sigma_branch, 4)

        self.loss = RLELoss(use_target_weight=False,
                            size_average=True,
                            residual=True,
                            q_dis='laplace')
        self.Initloss = RLELoss(use_target_weight=False,
                                size_average=True,
                                residual=True,
                                q_dis='laplace')
        self.initialize_weights()

        print("")

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # qqqq = get_2d_relative_pos_embed(self.pos1.shape[-1], self.Tsize * 8)

        pos2 = get_2d_sincos_pos_embed(self.pos2.shape[-1], self.Tsize * 4, cls_token=False)
        pos2 = pos2.reshape(self.Tsize[0] * 4, self.Tsize[1] * 4, self.numchannel//4*self.scalek)
        self.pos2.data.copy_(torch.from_numpy(pos2).float().unsqueeze(0))

        pos3 = get_2d_sincos_pos_embed(self.pos3.shape[-1], self.Tsize * 2, cls_token=False)
        pos3 = pos3.reshape(self.Tsize[0] * 2, self.Tsize[1] * 2, self.numchannel//2*self.scalek)
        self.pos3.data.copy_(torch.from_numpy(pos3).float().unsqueeze(0))

        pos4 = get_2d_sincos_pos_embed(self.pos4.shape[-1], self.Tsize, cls_token=False)
        pos4 = pos4.reshape(self.Tsize[0], self.Tsize[1], self.numchannel*self.scalek)
        self.pos4.data.copy_(torch.from_numpy(pos4).float().unsqueeze(0))


        device = torch.device("cuda")
        pcls_embed = PositionalEncoding(self.pcls_embed.shape[1], self.pcls_embed.shape[2], device)
        self.pcls_embed.data.copy_(pcls_embed.float().unsqueeze(0))

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        layer2, layer3, layer4 = self.backone(x)

        # layer4 = self.aifi(layer4)
        hotmap = None#self.out1(layer4)
        layer2, layer3, layer4 = self.head(layer2, layer3, layer4)


        b, c, h, w = layer4.shape
        pos2 = self.pos2.expand(b, -1, -1, -1).permute(0, 3, 1, 2)
        pos3 = self.pos3.expand(b, -1, -1, -1).permute(0, 3, 1, 2)
        pos4 = self.pos4.expand(b, -1, -1, -1).permute(0, 3, 1, 2)


        layer2 = self.bnF2(layer2)
        layer3 = self.bnF3(layer3)
        layer4 = self.bnF4(layer4)

        pos2 = self.convP2(pos2)
        pos3 = self.convP3(pos3)
        pos4 = self.convP4(pos4)


        b, c, h, w = layer4.shape
        h, w = self.TsizeS[0], self.TsizeS[1]
        ############sample############

        layer4_ = F.interpolate(layer4, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        pos4_ = F.interpolate(pos4, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        ############sample############

        sizes = [b, c, h, w]
        #################################################################
        inc = torch.cat([layer4_, torch.zeros((b, cfg.PointNms, self.nchannel)).cuda().float()], dim=1)
        pos = torch.cat([pos4_, self.pcls_embed.expand(b, -1, -1)], dim=1)

        inc = inc + pos
        xcs = []
        for cpb in self.InitEncoder:
            inc = cpb(inc)
            xcs.append(inc)
            inc = inc + pos
        output1 = self.linear12(F.relu(self.linear11(xcs[-1][:, -cfg.PointNms:, :])))
        #################################################################

        inint_c = get_cls_embed(layer2, layer3, layer4, output1[:, :, :2].detach(), sizes, self.wcl1)
        inint_p = get_cls_embed(pos2, pos3, pos4, output1[:, :, :2].detach(), sizes, self.wcl2)

        inint_c = inint_c[1] * self.wcl1[2] + inint_c[2] * self.wcl1[3] + inint_c[3] * self.wcl1[0]
        inint_p = inint_p[1] * self.wcl2[2] + inint_p[2] * self.wcl2[3] + inint_p[3] * self.wcl2[0]

        pclsp_embed = self.pcls_embed.expand(b, -1, -1) + inint_p * self.inintwp[0]


        hs = []
        inter_references = []
        xc_ = inint_c
        xc_ = xc_ #+ pclsp_embed

        reference_points = output1[:, :, :2].clip(0, 1).detach()
        init_reference = reference_points
        for i, cpb in enumerate(self.DEncoder):
            xc_ = cpb(xc_, pclsp_embed)
            tmp = self.fc_coord_branches[i](xc_)

            new_reference_points = tmp + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points

            inter_references.append(reference_points)
            hs.append(xc_)
            #################################
            inint_c = get_cls_embed(layer2, layer3, layer4, new_reference_points.detach(), sizes, self.wcl1)
            inint_p = get_cls_embed(pos2, pos3, pos4, new_reference_points.detach(), sizes, self.wcl2)

            inint_c = inint_c[1] * self.wcl1[2] + inint_c[2] * self.wcl1[3] + inint_c[3] * self.wcl1[0]
            inint_p = inint_p[1] * self.wcl2[2] + inint_p[2] * self.wcl2[3] + inint_p[3] * self.wcl2[0]

            pclsp_embed = self.pcls_embed.expand(b, -1, -1) + inint_p * self.inintwp[i+1]

            xc_ = xc_+ inint_c* self.inintwc[i+1]


            xc_ = xc_# + pclsp_embed

        outputs = []
        for lvl in range(len(inter_references)):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.fc_coord_branches[lvl](hs[lvl])
            tmp = tmp + reference

            outputs_sigma = self.fc_sigma_branches[lvl](hs[lvl])

            outputs_coord = tmp.sigmoid()
            delta_coord_output = self.fc_coord_output_branches[lvl](hs[lvl])
            outputs_coord = (outputs_coord + delta_coord_output)

            outputs_coord = torch.cat([outputs_coord, outputs_sigma], dim=-1)
            outputs.append(outputs_coord)

        return outputs#, output1, hotmap
