#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import cv2
import argparse
import torch
from thop import profile
import torch.nn as nn
from config.alignment import Alignment
import numpy as np

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.alignmentDataset import AlignmentDataset
from nme import NME
from thop import profile,clever_format

from nme import NME
from utils import accumulate_net, test_cal_eval
import config.config as cfg


def get_config(args):
    config = None
    config_name = args.config_name
    if config_name == "alignment":
        config = Alignment(args)
    else:
        assert NotImplementedError

    return config


def get_dataset(config, tsv_file, image_dir, loader_type, is_train):
    dataset = None
    if loader_type == "alignment":
        dataset = AlignmentDataset(
            tsv_file,
            image_dir,
            transforms.Compose([transforms.ToTensor()]),
            config.width,
            config.height,
            config.channels,
            config.means,
            config.scale,
            config.classes_num,
            config.crop_op,
            config.aug_prob,
            config.edge_info,
            config.flip_mapping,
            is_train,
            encoder_type=config.encoder_type
        )
    else:
        assert False
    return dataset


def get_dataloader(config, data_type):
    loader = None
    if data_type == "train":
        dataset = get_dataset(config, config.train_tsv_file, config.train_pic_dir, config.loader_type, is_train=True)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    elif data_type == "val":
        dataset = get_dataset(config, config.val_tsv_file, config.val_pic_dir, config.loader_type, is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.val_batch_size, num_workers=0)
    elif data_type == "test":
        dataset = get_dataset(config, config.test_tsv_file, config.test_pic_dir, config.loader_type, is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size, num_workers=0)
    else:
        assert False
    return loader


def cal_acc(key_points, gcoords,):

    diffv = key_points - gcoords
    errorv = np.sqrt(np.sum(np.power(diffv, 2), axis=-1))
    errorv = np.sum(errorv)
    #distance = gcoords[60, :] - gcoords[72, :]
    distance = gcoords[36,:]-gcoords[45,:]
    distance = np.sum(np.sqrt(np.sum(np.power(distance, 2), axis=-1)))

    ION_error = errorv / (cfg.PointNms * distance)



    eidl = np.array([36, 37, 38, 39, 40, 41])
    eidr = np.array([42, 43, 44, 45, 46, 47])
    left_center = (np.mean(gcoords[eidl], axis=0))
    right_center = (np.mean(gcoords[eidr], axis=0))
    distance = left_center- right_center
    #distance = gcoords[96, :] - gcoords[97, :]
    distance = np.sum(np.sqrt(np.sum(np.power(distance, 2), axis=-1)))

    IPN_error = errorv / (cfg.PointNms * distance)

    return IPN_error, ION_error


def model_initial(model, model_name):
    # 加载预训练模型
    pretrained_dict = torch.load(model_name)["model"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dictf = {k.replace('module.', ""): v for k, v in pretrained_dict.items() if k.replace('module.', "") in model_dict}
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dictf)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print("over")




def test(config):
    config.val_batch_size = 1
    test_loader = get_dataloader(config, "val")


    #from mobilenet.face_point_net_mobilenet import get_model
    # model_ = get_model()
    # model_name = "./save_model/mobilenet0.2939242.pth"
    # model_initial(model_, model_name)

    #from shuffleNet.face_point_net_shufflenet import get_model
    #from net.face_point_net_head import get_model
    # from net.ceph_regnet_refine import get_model
    # num_layers =18
    # head_conv = 256
    # heads = {'hm': 1, 'class': cfg.PointNms}
    # model_ = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    # model_name = "./save_model/mobilenet_wflw_0.041781746.pth"
    # model_initial(model_, model_name)
    # model_.cuda()
    # model_.eval()
    #
    # input_data = torch.rand((1, 3, 256,256)).cuda().float()
    # scripted_module = torch.jit.trace(model_.cuda().eval(), [input_data])
    # torch.jit.save(scripted_module, './save_model/mobilenet_wflw_0.041781746.pt')
    model = torch.jit.load('./save_model/ceph_resnet34_wflw_0.041287865.pt').cuda().float().eval()
    model.cuda()
    model.eval()



    ema_ION_error, FR, AUC, d_class_acc = test_cal_eval(model, NME, config, test_loader)

    print("nme = ", ema_ION_error, "FR = ", FR, "AUC = ", AUC, "\n")
    for key in d_class_acc:
        print(key, " = ", np.mean(d_class_acc[key]))
    print("over")


    # input = torch.randn(1, 3, 256, 256).cuda().float()
    # macs, params = profile(model_.cuda(), inputs=(input,))
    # gflops = macs * 2 / 1e9  # 转换为GFLOPs
    # params_mb = params * 4 / 1e6  # 转换为MB（float32）
    # print(gflops, params_mb)
    # macs, params = clever_format([macs, params], '%.3f')
    # print(macs, params)


    #
    # num = 0
    # IPN_errorS, ION_errorS = [], []
    # diffv = []
    # for iter, sample in enumerate(test_loader):
    #     test_data = sample["data"].cuda().float()
    #     label_coords = sample["label"][0].float()
    #
    #     with torch.no_grad():
    #         #outputs, inint_coords, h_ = model(test_data)
    #         outputs = model(test_data)
    #         outputs = torch.mean(torch.stack(outputs[-1:], dim=0), dim=0)
    #         pred = outputs[:, :, :2].squeeze().detach().cpu().numpy()
    #
    #         key_points = pred * np.array([[cfg.IMG_Height, cfg.IMG_Width]])  # cenp +cenp
    #
    #         label_coords = label_coords.squeeze().numpy() * np.array([[cfg.IMG_Height, cfg.IMG_Width]])  # cenp +cenp
    #         diffv.append(key_points - label_coords)
    #         IPN_error, ION_error = cal_acc(key_points, label_coords)
    #         IPN_errorS.append(IPN_error)
    #         ION_errorS.append(ION_error)
    #         num = num + 1
    #         print("nums = ", num)
    #
    # ION_errorS = np.array(ION_errorS)
    # IPN_errorS = np.array(IPN_errorS)
    # print("ION_error = ", np.mean(ION_errorS))
    # print("IPN_error = ", np.mean(IPN_errorS))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='key points')
    parser.add_argument('--exp_name', type=str, default='keyPoints', metavar='N', help='Name of the experiment')
    parser.add_argument("--image_dir", type=str, default="E:/DataSet/face_data/", help="the directory of image")
    parser.add_argument("--annot_dir", type=str, default="E:/DataSet/face_data/", help="the directory of annot")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument('--data_definition', type=str, default='WFLW', help="COFW, 300W, WFLW")
    parser.add_argument("--config_name", type=str, default="alignment", help="set configure file name")
    parser.add_argument("--batch_size", type=int, default=1, help="the batch size in train process")
    parser.add_argument('--width', type=int, default=256, help='the width of input image')
    parser.add_argument('--height', type=int, default=256, help='the height of input image')
    parser.add_argument('--epochs', type=int, default=301, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')  #
    parser.add_argument('--ema', type=bool, default=True, help='Use SGD')  #
    parser.add_argument('--lr', type=float, default=0.2 * 1e-3, metavar='LR',
                        help='learning rate ''(default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--model_ema_steps', type=float, default=1)
    parser.add_argument('--model_ema_decay', type=float, default=0.9998)
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()

    config = get_config(args)

    test(config)

