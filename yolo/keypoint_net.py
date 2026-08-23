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
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from timm.models.vision_transformer import Block
from functools import partial
from net.pos_embdb import get_2d_sincos_pos_embed, PositionalEncoding
from torch.nn.init import normal_
from net.regression_loss import RLELoss
from config import config as cfg
from yolo.model_parse import get_yolo11_model


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


def get_cls_embed(layer1, layer2, layer3, output, coords, sizes, wcl):
    coords = coords.view(-1, 1, cfg.PointNms, 2) * 2 - 1
    coords = coords.clip(-1, 1)
    cls1 = F.grid_sample(layer1, coords)
    cls2 = F.grid_sample(layer2, coords)
    # cls3 = F.grid_sample(layer3, coords)
    cls4 = F.grid_sample(output, coords)

    cls1 = cls1.squeeze(dim=2).permute(0, 2, 1)
    cls2 = cls2.squeeze(dim=2).permute(0, 2, 1)
    # cls3 = cls3.squeeze(dim=2).permute(0, 2, 1)
    cls4 = cls4.squeeze(dim=2).permute(0, 2, 1)

    cls = cls1 * wcl[1] + cls2 * wcl[2]  + cls4 * wcl[0] #+ cls3 * wcl[3]
    # cls = torch.cat([cls1, cls2, cls3, cls4], dim=1)
    return cls


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


BN_MOMENTUM = 0.1
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 16, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


class FResNet(nn.Module):

    def __init__(self, num_layers, heads, head_conv):
        super(FResNet, self).__init__()
        block, layers = resnet_spec[num_layers]
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(FResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        print("")

        # used for deconv layers

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)

        x = self.maxpool(conv1)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return layer1, layer2, layer3, layer4

    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> init deconv weights from normal distribution')


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


class get_model_yolo(nn.Module):

    def __init__(self, model_confile):
        super(get_model_yolo, self).__init__()
        self.inchannel = 768
        self.outchannel = 512
        model = get_yolo11_model(cfg=model_confile)
        self.backone = model

        self.out1 = nn.Conv2d(in_channels=self.inchannel, out_channels=1, kernel_size=1, stride=1)

        self.Tsize = torch.tensor([8, 8]).int()
        self.TsizeS = self.Tsize * 2
        self.pcls_embed = nn.Parameter(torch.zeros(1, cfg.PointNms, self.outchannel), requires_grad=False)

        self.convF1 = nn.Conv2d(in_channels=384, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convF2 = nn.Conv2d(in_channels=768, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convF3 = nn.Conv2d(in_channels=768, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convF4 = nn.Conv2d(in_channels=768, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.bnF1 = nn.BatchNorm2d(self.outchannel, momentum=BN_MOMENTUM)
        self.bnF2 = nn.BatchNorm2d(self.outchannel, momentum=BN_MOMENTUM)
        self.bnF3 = nn.BatchNorm2d(self.outchannel, momentum=BN_MOMENTUM)
        self.bnF4 = nn.BatchNorm2d(self.outchannel, momentum=BN_MOMENTUM)
        self.convP1 = nn.Conv2d(in_channels=384, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convP2 = nn.Conv2d(in_channels=768, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convP3 = nn.Conv2d(in_channels=768, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convP4 = nn.Conv2d(in_channels=768, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)

        self.inintwc = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]), requires_grad=True)
        self.inintwp = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]), requires_grad=True)
        self.pg = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wg = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wcl1 = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wcl2 = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.pos1 = nn.Parameter(torch.zeros(1, self.Tsize[0] * 4, self.Tsize[1] * 4, self.inchannel // 2),
                                 requires_grad=False)
        self.pos2 = nn.Parameter(torch.zeros(1, self.Tsize[0] * 2, self.Tsize[1] * 2, self.inchannel // 1),
                                 requires_grad=False)
        self.pos3 = nn.Parameter(torch.zeros(1, self.Tsize[0] * 1, self.Tsize[1] * 1, self.inchannel // 1),
                                 requires_grad=False)
        self.pos4 = nn.Parameter(torch.zeros(1, self.Tsize[0], self.Tsize[1], self.inchannel),
                                 requires_grad=False)  # fixed sin-cos embedding
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.InitEncoder = nn.ModuleList([
            Block(dim=self.outchannel, num_heads=self.outchannel // 64, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
            for i in range(4)])
        self.linear11 = nn.Linear(self.outchannel, 256)
        self.linear12 = nn.Linear(256, 4)

        self.DEncoder = nn.ModuleList([
            Block(dim=self.outchannel, num_heads=self.outchannel // 64, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
            for i in range(4)])

        # self.linear21 = nn.Linear(numchannel, 256)
        # self.linear22 = nn.Linear(256, 4)

        fc_coord_branch = []
        for _ in range(2):
            fc_coord_branch.append(nn.Linear(self.outchannel, self.outchannel))
            fc_coord_branch.append(nn.ReLU())
        fc_coord_branch.append(nn.Linear(self.outchannel, 2))
        fc_coord_branch = nn.Sequential(*fc_coord_branch)
        self.fc_coord_branches = self._get_clones(fc_coord_branch, 4)
        self.fc_coord_output_branches = self._get_clones(fc_coord_branch, 4)

        fc_sigma_branch = []
        for _ in range(2):
            fc_sigma_branch.append(nn.Linear(self.outchannel, self.outchannel))
        fc_sigma_branch.append(Linear_with_norm(self.outchannel, 2, norm=False))
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
        pos1 = get_2d_sincos_pos_embed(self.pos1.shape[-1], self.Tsize * 4, cls_token=False)
        pos1 = pos1.reshape(self.Tsize[0] * 4, self.Tsize[1] * 4, self.inchannel // 2)
        self.pos1.data.copy_(torch.from_numpy(pos1).float().unsqueeze(0))

        pos2 = get_2d_sincos_pos_embed(self.pos2.shape[-1], self.Tsize * 2, cls_token=False)
        pos2 = pos2.reshape(self.Tsize[0] * 2, self.Tsize[1] * 2, self.inchannel // 1)
        self.pos2.data.copy_(torch.from_numpy(pos2).float().unsqueeze(0))

        pos3 = get_2d_sincos_pos_embed(self.pos3.shape[-1], self.Tsize * 1, cls_token=False)
        pos3 = pos3.reshape(self.Tsize[0] * 1, self.Tsize[1] * 1, self.inchannel // 1)
        self.pos3.data.copy_(torch.from_numpy(pos3).float().unsqueeze(0))

        pos4 = get_2d_sincos_pos_embed(self.pos4.shape[-1], self.Tsize, cls_token=False)
        pos4 = pos4.reshape(self.Tsize[0], self.Tsize[1], self.inchannel)
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
        layer1, layer2, layer4, layer3 = self.backone(x)
        hotmap = self.out1(layer4)

        b, c, h, w = layer4.shape
        pos1 = self.pos1.expand(b, -1, -1, -1).permute(0, 3, 1, 2)
        pos2 = self.pos2.expand(b, -1, -1, -1).permute(0, 3, 1, 2)
        pos3 = self.pos3.expand(b, -1, -1, -1).permute(0, 3, 1, 2)
        pos4 = self.pos4.expand(b, -1, -1, -1).permute(0, 3, 1, 2)

        layer1 = self.bnF1(self.convF1(layer1))
        layer2 = self.bnF2(self.convF2(layer2))
        # layer3 = self.bnF3(self.convF3(layer3))
        layer4 = self.bnF4(self.convF4(layer4))

        pos1 = self.convP1(pos1)
        pos2 = self.convP2(pos2)
        # pos3 = self.convP3(pos3)
        pos4 = self.convP4(pos4)


        b, c, h, w = layer4.shape
        h, w = self.TsizeS[0], self.TsizeS[1]
        ############sample############
        # layer1_ = F.interpolate(layer1, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        # layer2_ = F.interpolate(layer2, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        # layer3_ = F.interpolate(layer3, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        layer4_ = F.interpolate(layer4, (self.TsizeS[0], self.TsizeS[1]), mode='bilinear', align_corners=True).view(b, c, h * w).permute(0, 2, 1)

        # pos1_ = F.interpolate(pos1, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        # pos2_ = F.interpolate(pos2, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        # pos3_ = F.interpolate(pos3, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        pos4_ = F.interpolate(pos4, (self.TsizeS[0], self.TsizeS[1]), mode='bilinear', align_corners=True).view(b, c, h * w).permute(0, 2, 1)
        ############sample############

        # layer1 = layer1 + pos1
        # layer2 = layer2 + pos2
        # layer3 = layer3 + pos3
        # layer4 = layer4 + pos4

        sizes = [b, c, h, w]
        #################################################################
        inc = torch.cat([layer4_, torch.zeros((b, cfg.PointNms, self.outchannel)).cuda().float()], dim=1)
        pos = torch.cat([pos4_, self.pcls_embed.expand(b, -1, -1)], dim=1)

        inc = inc + pos
        xcs = []
        for cpb in self.InitEncoder:
            inc = cpb(inc)
            xcs.append(inc)
            inc = inc + pos
        output1 = self.linear12(F.relu(self.linear11(xcs[-1][:, -cfg.PointNms:, :])))
        #################################################################

        inint_c = get_cls_embed(layer1, layer2, layer3, layer4, output1[:, :, :2].detach(), sizes, self.wcl1)
        inint_p = get_cls_embed(pos1, pos2, pos3, pos4, output1[:, :, :2].detach(), sizes, self.wcl2)

        pclsc_embed = inint_c
        zz = torch.zeros((b, cfg.PointNms * 3, 512)).cuda().float()
        pclsp_embed = self.pcls_embed.expand(b, -1, -1) + inint_p * self.inintwp[0]

        hs = []
        inter_references = []
        xc_ = pclsc_embed
        pos_ = pclsp_embed
        xc_ = xc_ + pos_

        reference_points = output1[:, :, :2].detach().clip(0, 1)
        init_reference = reference_points
        for i, cpb in enumerate(self.DEncoder):
            xc_ = cpb(xc_)
            tmp = self.fc_coord_branches[i](xc_[:, -cfg.PointNms:, :])

            new_reference_points = tmp + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points

            inter_references.append(reference_points)
            hs.append(xc_[:, -cfg.PointNms:, :])
            #################################
            inint_c = get_cls_embed(layer1, layer2, layer3, layer4, new_reference_points.detach(), sizes, self.wcl1)
            inint_p = get_cls_embed(pos1, pos2, pos3, pos4, new_reference_points.detach(), sizes, self.wcl2)

            pclsc_embed = inint_c
            pclsp_embed = self.pcls_embed.expand(b, -1, -1) + inint_p * self.inintwp[i + 1]

            pos_ = pclsp_embed
            xc_ = xc_ + pclsc_embed * self.inintwc[i + 1]
            xc_ = xc_ + pos_

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

        return outputs, output1, hotmap
