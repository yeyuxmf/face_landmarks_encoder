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
import logging
import functools

import numpy as np
import copy
import torch
import torch.nn as nn
import torch._utils
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from timm.models.vision_transformer import Block
from functools import partial
import torch.nn.functional as F
from net.pos_embdb import get_2d_sincos_pos_embed, PositionalEncoding
from torch.nn.init import normal_
from net.regression_loss import RLELoss
from config import config as cfg
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)



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


def get_cls_embed(layer1, layer2, layer3, output, coords, sizes, wcl):

    coords = coords.view(-1, 1, cfg.PointNms, 2) * 2 - 1
    coords = coords.clip(-1, 1)
    cls1 = F.grid_sample(layer1, coords)
    cls2 = F.grid_sample(layer2, coords)
    cls3 = F.grid_sample(layer3, coords)
    cls4 = F.grid_sample(output, coords)

    cls1 = cls1.squeeze(dim=2).permute(0, 2, 1)
    cls2 = cls2.squeeze(dim=2).permute(0, 2, 1)
    cls3 = cls3.squeeze(dim=2).permute(0, 2, 1)
    cls4 = cls4.squeeze(dim=2).permute(0, 2, 1)

    cls = cls1 * wcl[1] + cls2 * wcl[2] + cls3 * wcl[3] + cls4 * wcl[0]

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


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i],
                                       momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(HighResolutionNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg['MODEL']['EXTRA']['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # Classification Head
        self.incre_modules, self.downsamp_modules, \
        self.final_layer = self._make_head(pre_stage_channels)

        self.classifier = nn.Linear(2048, 1000)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # output Head
        output = []
        y = self.incre_modules[0](y_list[0])
        output.append(y)
        for i in range(len(self.downsamp_modules)):
            down = self.downsamp_modules[i](y)
            y = self.incre_modules[i + 1](y_list[i + 1])+ down
            output.append(y)

        return output

    def init_weights(self, pretrained='', ):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            print(pretrained)
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(pretrained,**kwargs):
    import yaml
    from yaml.loader import SafeLoader

    # Open the file and load the file
    with open("./net/cls_hrnet_w18.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    model = HighResolutionNet(config, **kwargs)
    model.init_weights(pretrained=pretrained)
    return model

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


class get_model(nn.Module):

    def __init__(self, num_layers, heads, head_conv=256):
        super(get_model, self).__init__()

        self.nchannel = 1024
        self.outchannel = 512
        model = get_cls_net(pretrained="./net/hrnetv2_w18_imagenet_pretrained.pth")
        self.backone = model

        self.out1 = nn.Conv2d(in_channels= self.nchannel, out_channels=1, kernel_size=1, stride=1)

        self.Tsize = torch.tensor([8, 8]).int()
        self.TsizeS = self.Tsize*2
        self.pcls_embed = nn.Parameter(torch.zeros(1, cfg.PointNms, self.outchannel), requires_grad=False)

        self.convF1 = nn.Conv2d(in_channels=self.nchannel//8, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convF2 = nn.Conv2d(in_channels=self.nchannel//4, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convF3 = nn.Conv2d(in_channels=self.nchannel//2, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convF4 = nn.Conv2d(in_channels=self.nchannel, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.bnF1 = nn.BatchNorm2d(self.outchannel, momentum=BN_MOMENTUM)
        self.bnF2 = nn.BatchNorm2d(self.outchannel, momentum=BN_MOMENTUM)
        self.bnF3 = nn.BatchNorm2d(self.outchannel, momentum=BN_MOMENTUM)
        self.bnF4 = nn.BatchNorm2d(self.outchannel, momentum=BN_MOMENTUM)
        self.convP1 = nn.Conv2d(in_channels=self.nchannel//8, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convP2 = nn.Conv2d(in_channels=self.nchannel//4, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convP3 = nn.Conv2d(in_channels=self.nchannel//2, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)
        self.convP4 = nn.Conv2d(in_channels=self.nchannel, out_channels=self.outchannel, kernel_size=1, stride=1, bias=False)

        self.inintwc = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]), requires_grad=True)
        self.inintwp = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]), requires_grad=True)
        self.pg = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wg = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wcl1 = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.wcl2 = nn.Parameter(torch.tensor([1, 0.8, 0.6, 0.4]), requires_grad=True)
        self.pos1 = nn.Parameter(torch.zeros(1, self.Tsize[0] * 8, self.Tsize[1] * 8,  self.nchannel // 8), requires_grad=False)
        self.pos2 = nn.Parameter(torch.zeros(1, self.Tsize[0] * 4, self.Tsize[1] * 4,  self.nchannel // 4),requires_grad=False)
        self.pos3 = nn.Parameter(torch.zeros(1, self.Tsize[0] * 2, self.Tsize[1] * 2,  self.nchannel // 2),requires_grad=False)
        self.pos4 = nn.Parameter(torch.zeros(1, self.Tsize[0],  self.Tsize[1],  self.nchannel),requires_grad=False)  # fixed sin-cos embedding
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
        pos1 = get_2d_sincos_pos_embed(self.pos1.shape[-1], self.Tsize * 8, cls_token=False)
        pos1 = pos1.reshape(self.Tsize[0] * 8, self.Tsize[1] * 8,  self.nchannel//8)
        self.pos1.data.copy_(torch.from_numpy(pos1).float().unsqueeze(0))

        pos2 = get_2d_sincos_pos_embed(self.pos2.shape[-1], self.Tsize * 4, cls_token=False)
        pos2 = pos2.reshape(self.Tsize[0] * 4, self.Tsize[1] * 4, self.nchannel//4)
        self.pos2.data.copy_(torch.from_numpy(pos2).float().unsqueeze(0))

        pos3 = get_2d_sincos_pos_embed(self.pos3.shape[-1], self.Tsize * 2, cls_token=False)
        pos3 = pos3.reshape(self.Tsize[0] * 2, self.Tsize[1] * 2, self.nchannel//2)
        self.pos3.data.copy_(torch.from_numpy(pos3).float().unsqueeze(0))

        pos4 = get_2d_sincos_pos_embed(self.pos4.shape[-1], self.Tsize, cls_token=False)
        pos4 = pos4.reshape(self.Tsize[0], self.Tsize[1], self.nchannel)
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
        layer1, layer2, layer3, layer4 = self.backone(x)
        hotmap = self.out1(layer4)

        b, c, h, w = layer4.shape
        pos1 = self.pos1.expand(b, -1, -1, -1).permute(0, 3, 1, 2)
        pos2 = self.pos2.expand(b, -1, -1, -1).permute(0, 3, 1, 2)
        pos3 = self.pos3.expand(b, -1, -1, -1).permute(0, 3, 1, 2)
        pos4 = self.pos4.expand(b, -1, -1, -1).permute(0, 3, 1, 2)

        layer1 = self.bnF1(self.convF1(layer1))
        layer2 = self.bnF2(self.convF2(layer2))
        layer3 = self.bnF3(self.convF3(layer3))
        layer4 = self.bnF4(self.convF4(layer4))

        pos1 = self.convP1(pos1)
        pos2 = self.convP2(pos2)
        pos3 = self.convP3(pos3)
        pos4 = self.convP4(pos4)
        b, c, h, w = layer4.shape
        h, w = self.TsizeS[0], self.TsizeS[1]
        ############sample############
        # layer1_ = F.interpolate(layer1, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        # layer2_ = F.interpolate(layer2, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        # layer3_ = F.interpolate(layer3, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        layer4_ = F.interpolate(layer4, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)

        # pos1_ = F.interpolate(pos1, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        # pos2_ = F.interpolate(pos2, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        # pos3_ = F.interpolate(pos3, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
        pos4_ = F.interpolate(pos4, (self.TsizeS[0], self.TsizeS[1])).view(b, c, h * w).permute(0, 2, 1)
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
            tmp = self.fc_coord_branches[i](xc_)

            new_reference_points = tmp + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points

            inter_references.append(reference_points)
            hs.append(xc_)
            #################################
            inint_c = get_cls_embed(layer1, layer2, layer3, layer4, new_reference_points.detach(), sizes, self.wcl1)
            inint_p = get_cls_embed(pos1, pos2, pos3, pos4, new_reference_points.detach(), sizes, self.wcl2)

            pclsc_embed = inint_c
            pclsp_embed = self.pcls_embed.expand(b, -1, -1) + inint_p * self.inintwp[i+1]

            pos_ = pclsp_embed
            xc_ = xc_ + pclsc_embed* self.inintwc[i+1]
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
