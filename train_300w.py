#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import cv2
import argparse
import torch
from thop import profile,clever_format
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.load_train_300w import TrainData
from data.load_test_300w import TestData
from torch.utils.data import DataLoader
from config.alignment import Alignment
#from net.ceph_regnet_refine import get_model
from net.face_point_net_head import get_model
#from shuffleNet.face_point_net_shufflenet import get_model
#from mobilenet.face_point_net_mobilenet import get_model
from data.alignmentDataset import AlignmentDataset
from nme import NME
from c_adamw import AdamW as C_AdamW
import config.config as cfg
from utils import accumulate_net, cal_eval
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
        dataset = get_dataset(config,config.train_tsv_file,config.train_pic_dir,config.loader_type,is_train=True)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,num_workers=0)
    elif data_type == "val":
        dataset = get_dataset(config,config.val_tsv_file,config.val_pic_dir,config.loader_type,is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.val_batch_size,num_workers=0)
    elif data_type == "test":
        dataset = get_dataset(config,config.test_tsv_file,config.test_pic_dir,config.loader_type,is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size,num_workers=0)
    else:
        assert False
    return loader


def cal_acc(key_points, gcoords,):

    diffv = key_points - gcoords
    errorv = np.sqrt(np.sum(np.power(diffv, 2), axis=-1))
    errorv = np.sum(errorv)

    distance = gcoords[36,:]-gcoords[45,:]
    distance = np.sum(np.sqrt(np.sum(np.power(distance, 2), axis=-1)))

    ION_error = errorv / (cfg.PointNms * distance)

    eidl = np.array([36, 37, 38, 39, 40, 41])
    eidr = np.array([42, 43, 44, 45, 46, 47])
    left_center = (np.mean(gcoords[eidl], axis=0))
    right_center = (np.mean(gcoords[eidr], axis=0))
    distance = left_center- right_center
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


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('./outputs/' + args.exp_name):
        os.makedirs('./outputs/' + args.exp_name)
    if not os.path.exists('./outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('./outputs/' + args.exp_name + '/' + 'models')
    os.system('cp main_cls.py outputs' + '/' + args.exp_name + '/' + 'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

def train(args, io, config):
    custom_optim="c_adamw"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader(config, "train")
    test_loader = get_dataloader(config, "val")

    # Try to load models
    num_layers =18
    head_conv = 256
    heads = {'hm': 1, 'class': cfg.PointNms}
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    # model_name = "./outputs/emabest2.pth"
    # model_initial(model, model_name)
    if args.ema:
        model_ema = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv).cuda().eval()

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.use_sgd:
        print("Use SGD")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # if custom_optim == "c_adamw":
    #     opt = C_AdamW(model.parameters(), betas = (0.9, 0.95), weight_decay= 1e-4, lr = args.lr)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=0.5e-6, last_epoch = -1)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    model.cuda()
    model.train()

    # if args.ema:
    #     adjust = args.batch_size * args.model_ema_steps / args.epochs
    #     alpha = 1.0 - args.model_ema_decay
    #     alpha = min(1.0, alpha * adjust)
    #     model_ema = ExponentialMovingAverage(model_ema, device=device, decay=1.0 - alpha)

    input = torch.randn(1, 3, 256, 256).cuda().float()
    macs, params = profile(model.to(device), inputs=(input,))

    macs, params = clever_format([macs, params], '%.3f')
    print(macs, params)

    scaler = torch.amp.GradScaler()
    inter_nums = len(train_loader)
    total_acc = 1000
    TMAXcounts = None
    for epoch in range(0, args.epochs):
        ####################
        # Train
        ####################

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        if epoch <0:
            continue

        train_loss = 0.0
        inint_loss = 0.0
        loss_ml = 0.0
        loss_dl = 0.0
        edgeloss = 0
        # for data, edges, label in train_loader:
        tic = time.time()
        nums = 0
        model.train()
        tic = time.time()
        for iter, sample in enumerate(train_loader):
            train_data = sample["data"]
            label_re = sample["label"][0]
            train_data = train_data.cuda().float()
            label_re = label_re.cuda().float()
            pointmap = sample["pointmap"].cuda().float()

            nums = nums +1
            opt.zero_grad()
            try:
                with torch.amp.autocast(device_type='cuda'):

                    outputs, inint_coords, prehotmap = model(train_data)
                    inint_loss_ = model.Initloss(inint_coords, label_re)
                    loss1 = inint_loss_#model.loss(outputs[0], label_re)
                    loss2 = inint_loss_#model.loss(outputs[1], label_re)
                    loss3 = inint_loss_#model.loss(outputs[2], label_re)
                    loss4 = loss3#model.loss(outputs[3], label_re)
                    loss = (loss1 + loss2 + loss3)# + loss4)

                    # loss_ml_, loss_dl_, loss_re_, loss_fl_ = rcal_loss(prehotmap, pointmap)
                    # edgeloss_ =  loss_fl_# edge_loss(outputs, label_re.repeat(1, 4, 1))
                    loss_ml_, loss_dl_ = loss1, loss1
                    loss = inint_loss_# +loss # + loss_ml_+loss_dl_#+ loss_ml_+ loss_dl_ #+ edgeloss_ +

                    scaler.scale(loss).backward()
                    # Unscales gradients and calls
                    # or skips optimizer.step()
                    scaler.step(opt)
                    # Updates the scale for next iteration
                    scaler.update()
                if model_ema is not None and epoch% args.model_ema_steps == 0:
                    accumulate_net(model_ema, model, 0.5 ** (config.batch_size / 10000.0))
                # if model_ema and epoch % args.model_ema_steps == 0:
                #     model_ema.update_parameters(model)
                train_loss += loss.item()
                inint_loss += inint_loss_.item()
                loss_ml += loss_ml_.item()
                loss_dl += loss_dl_.item()
            except Exception :
                print("")
            # edgeloss += edgeloss_.item()
            if nums % cfg.VIEW_NUMS == 0:
                toc = time.time()
                train_loss = train_loss/ (cfg.VIEW_NUMS)
                inint_loss = inint_loss/ (cfg.VIEW_NUMS)
                loss_ml = loss_ml/ (cfg.VIEW_NUMS)
                loss_dl = loss_dl/ (cfg.VIEW_NUMS)
                # edgeloss = edgeloss/ (cfg.VIEW_NUMS)

                print("lr = ", opt.param_groups[0]['lr'])#, "loss1 = ", loss1.item(), "loss2 = ", loss2.item(), "loss3 = ", loss3.item(), "loss4 = ", loss4.item())
                outstr = 'epoch %d /%d,epoch %d /%d, loss: %.6f, inint_loss: %.6f, loss_ml: %.6f, loss_dl: %.6f, const time: %.6f' % (
                 epoch,args.epochs, nums, inter_nums, train_loss, inint_loss, loss_ml, loss_dl, toc - tic)

                io.cprint(outstr)
                train_loss = 0.0
                inint_loss = 0.0
                loss_ml = 0.0
                loss_dl = 0.0
                # edgeloss = 0.0
                tic = time.time()
        if 0 == epoch % 1 and epoch>=0:

            model_ION_error = cal_eval(model, NME, config, test_loader)
            ema_ION_error = cal_eval(model_ema, NME, config, test_loader)
            ION_error = min(ema_ION_error, model_ION_error)
            if total_acc > ION_error:
                total_acc = ION_error
                if ema_ION_error<model_ION_error:
                   torch.save({'model': model_ema.state_dict(), 'epoch': epoch},'outputs/' + 'emabest4.pth')
                else:
                    torch.save({'model': model.state_dict(), 'epoch': epoch}, 'outputs/' + 'best4.pth')
            print("best ION_error = ", total_acc)
            print("model_ION_error = ", model_ION_error)
            print("ema_ION_error = ", ema_ION_error)


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    # Training settings
    parser = argparse.ArgumentParser(description='key points')
    parser.add_argument('--exp_name', type=str, default='keyPoints', metavar='N',help='Name of the experiment')
    parser.add_argument("--image_dir", type=str, default="E:/DataSet/face_data/", help="the directory of image")
    parser.add_argument("--annot_dir", type=str, default="E:/DataSet/face_data/", help="the directory of annot")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument('--data_definition', type=str, default='WFLW', help="COFW, 300W, WFLW")
    parser.add_argument("--config_name", type=str, default="alignment", help="set configure file name")
    parser.add_argument("--batch_size", type=int, default=16, help="the batch size in train process")
    parser.add_argument('--width', type=int, default=256, help='the width of input image')
    parser.add_argument('--height', type=int, default=256, help='the height of input image')
    parser.add_argument('--epochs', type=int, default=301, metavar='N',help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')#
    parser.add_argument('--ema', type=bool, default=True, help='Use SGD')#
    parser.add_argument('--lr', type=float, default= 0.15*1e-3, metavar='LR',
                        help='learning rate ''(default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--model_ema_steps', type=float, default= 1)
    parser.add_argument('--model_ema_decay', type=float, default= 0.9998)
    parser.add_argument('--seed', type=int, default=42, metavar='S',help='random seed (default: 1)')



    args = parser.parse_args()
    _init_()
    config = get_config(args)


    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')



    train(args, io, config)

