import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from nme import FR_AUC

from config import config as cfg
def accumulate_net(model1, model2, decay):
    """
        operation: model1 = model1 * decay + model2 * (1 - decay)
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(
            other=par2[k].data.to(par1[k].data.device),
            alpha=1 - decay)

    par1 = dict(model1.named_buffers())
    par2 = dict(model2.named_buffers())
    for k in par1.keys():
        if par1[k].data.is_floating_point():
            par1[k].data.mul_(decay).add_(
                other=par2[k].data.to(par1[k].data.device),
                alpha=1 - decay)
        else:
            par1[k].data = par2[k].data.to(par1[k].data.device)
def cal_eval(model, NME, config, test_loader):
    model.eval()
    mask = []
    list_nmes = [[] for i in range(1)]
    metric_nme = NME(nme_left_index=config.nme_left_index, nme_right_index=config.nme_right_index)
    for iter, sample in enumerate(test_loader):
        test_data = sample["data"].cuda().float()
        labels = sample["label"][0].cuda().float()

        for i in range(len(sample["image_path"])):
            flag = "ibug" in sample["image_path"][i]
            mask.append(flag)
        with torch.no_grad():
            outputs, inint_coords, prehotmap = model(test_data)
        outputs =[inint_coords]
            # metrics
        outputs = torch.mean(torch.stack(outputs[-1:], dim=0), dim=0)
        nume = metric_nme.test(outputs[:, :, :2], labels)
        list_nmes[0] += nume

    ION_error = [np.mean(nmes) for nmes in list_nmes]

    ION_error = np.mean(np.array(ION_error))

    return ION_error


def get_different_class_acc(data_definition):
    d_class = {}
    class_name_key = {}
    if "300W" == data_definition:
        d_class = {"challengetest":[], "commontest": []}
        file_ = open("E:/DataSet/face_data/300w//list_challengetest.txt", "r")
        for line in file_.readlines():
            d_class["challengetest"].append(os.path.basename(line.strip().split(",")[0]))
        file_ = open("E:/DataSet/face_data/300w/list_commontest.txt", "r")
        for line in file_.readlines():
            d_class["commontest"].append(os.path.basename(line.strip().split(",")[0]))
    elif "WFLW" == data_definition:
        d_class = {"largepose": [], "expression": [], "illumination": [], "makeup": [], "occlusion": [], "blur": []}

        tsv_flie = "E:/DataSet/face_data/WFLW/test_blur_metadata.tsv"
        items = pd.read_csv(tsv_flie, sep="\t")
        for index in range(len(items)):
            d_class["blur"].append(os.path.basename(items.iloc[index, 0]))
        tsv_flie = "E:/DataSet/face_data/WFLW/test_expression_metadata.tsv"
        items = pd.read_csv(tsv_flie, sep="\t")
        for index in range(len(items)):
            d_class["expression"].append(os.path.basename(items.iloc[index, 0]))
        tsv_flie = "E:/DataSet/face_data/WFLW/test_illumination_metadata.tsv"
        items = pd.read_csv(tsv_flie, sep="\t")
        for index in range(len(items)):
            d_class["illumination"].append(os.path.basename(items.iloc[index, 0]))
        tsv_flie = "E:/DataSet/face_data/WFLW/test_largepose_metadata.tsv"
        items = pd.read_csv(tsv_flie, sep="\t")
        for index in range(len(items)):
            d_class["largepose"].append(os.path.basename(items.iloc[index, 0]))
        tsv_flie = "E:/DataSet/face_data/WFLW/test_makeup_metadata.tsv"
        items = pd.read_csv(tsv_flie, sep="\t")
        for index in range(len(items)):
            d_class["makeup"].append(os.path.basename(items.iloc[index, 0]))
        tsv_flie = "E:/DataSet/face_data/WFLW/test_occlusion_metadata.tsv"
        items = pd.read_csv(tsv_flie, sep="\t")
        for index in range(len(items)):
            d_class["occlusion"].append(os.path.basename(items.iloc[index, 0]))

    return d_class
def find_keys_for_value(target_value, data_dict):
    """查找值属于哪些键"""
    result = []
    for key, value_list in data_dict.items():
        if target_value in value_list:
            result.append(key)
    return result
def test_cal_eval(model, NME, config, test_loader):
    model.eval()
    mask = []
    list_nmes = [[] for i in range(1)]
    metric_nme = NME(nme_left_index=config.nme_left_index, nme_right_index=config.nme_right_index)
    metric_fr_auc = FR_AUC(data_definition=config.data_definition)

    d_class_name = get_different_class_acc(config.data_definition)
    d_class_acc = {key: [] for key in d_class_name}

    save_root = "./save_img_view/"
    for iter, sample in enumerate(test_loader):
        print("iter  ", iter)
        test_data = sample["data"].cuda().float()
        labels = sample["label"][0].cuda().float()
        data_name = os.path.basename(sample["image_path"][0])
        for i in range(len(sample["image_path"])):
            flag = "ibug" in sample["image_path"][i]
            mask.append(flag)
        with torch.no_grad():
            outputs = model(test_data)

            # metrics
        outputs = torch.mean(torch.stack(outputs[-1:], dim=0), dim=0)
        nume = metric_nme.test(outputs[:, :, :2], labels)
        list_nmes[0] += nume

        keys = find_keys_for_value(data_name, d_class_name)
        for ki in range(len(keys)):
            d_class_acc[keys[ki]].append(nume[0])

            if not os.path.exists(save_root + keys[ki]+"/ceph/"):
                os.makedirs(save_root + keys[ki]+"/ceph/")
            img = sample["rowimg"].squeeze().detach().cpu().numpy()
            landmarks = (outputs[:, :, :2].squeeze().detach().cpu().numpy()*256).astype(np.int32)
            gt_numpy =(labels.squeeze().detach().cpu().numpy()*256).astype(np.int32)
            for pi in range(landmarks.shape[0]):
                coord = landmarks[pi].astype(np.int32)
                gt = gt_numpy[pi].astype(np.int32)
                img = cv2.circle(img, (coord[0], coord[1]), 1, (255, 0, 0), thickness=2)
                img = cv2.circle(img, (gt[0], gt[1]), 1, (0, 0, 255), thickness=2)
                # img = cv2.putText(img, str(pi), (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),1)
            cv2.imwrite(save_root + keys[ki]+"/ceph/" +data_name+"_mo_"+str(nume[0])+".jpg", img)

    ION_error = [[np.mean(nmes)]+ metric_fr_auc.test(nmes) for nmes in list_nmes]


    ION_error, FR, AUC = ION_error[0][0], ION_error[0][1], ION_error[0][2]

    return ION_error, FR, AUC, d_class_acc