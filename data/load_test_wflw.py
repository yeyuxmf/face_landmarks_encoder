import os
import random
import cv2
import numpy as np
from math import radians, cos, sin
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import imutils
import torch
from PIL import Image

import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from config import config as cfg

class Transforms():
    def __init__(self):
        pass

    def resize(self, image, landmarks, img_size):
        # 调整图像大小
        image = TF.resize(image, img_size)
        return image, landmarks



    def crop_face(self, image, landmarks, crops):
        # 获取裁剪参数
        x1 = np.min(landmarks[:, 0], axis=0)#int(crops['left'])
        y1 = np.min(landmarks[:, 1], axis=0)#int(crops['top'])
        x2 = np.max(landmarks[:, 0], axis=0)#int(crops['left'])
        y2 = np.max(landmarks[:, 1], axis=0)#int(crops['top'])

        # image_= np.array(image)
        # for pi in range(landmarks.shape[0]):
        #     image_ = cv2.rectangle(image_, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        # for pi in range(landmarks.shape[0]):
        #     coord = landmarks[pi].astype(np.int32)
        #     image_ = cv2.circle(image_, (coord[0], coord[1]), 1, (255, 0, 0) , thickness=2)
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", image_)
        # cv2.waitKey(0)


        height, width, _ = np.array(image).shape
        cwidth, cheight = x2-x1, y2-y1
        sacle = 0.1#np.random.randint(5, 30, 1)[0] / 100.0
        x1, x2 = int(max(0, x1 - sacle*cwidth)),int( min(x2 + sacle*cwidth, width))
        y1, y2 = int(max(0, y1 - sacle*cheight)), int(min(y2 + sacle*cheight, height))

        # 对图像进行裁剪
        image = np.array(image)
        image = Image.fromarray((image[y1: y2, x1: x2, :]).copy())

        # 获取裁剪后的图像形状
        img_shape = np.array(image).shape
        # 对关键点坐标进行裁剪后的调整
        landmarks = torch.tensor(landmarks) - torch.tensor([[x1, y1]])
        # 归一化关键点坐标
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        # 将图像从数组转换为 PIL 图像对象
        image = Image.fromarray(image)
        # 裁剪图像并调整关键点
        image, landmarks = self.crop_face(image, landmarks, crops)
        # 调整图像大小
        image, landmarks = self.resize(image, landmarks, (cfg.IMG_Width, cfg.IMG_Height))

        # 将图像从 PIL 图像对象转换为 Torch 张量
        image = TF.to_tensor(image)
        # 标准化图像像素值
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks





class TestData(Dataset):
    def __init__(self, file_path, data_root):
        # 解析 XML 文件


        # 初始化变量
        self.data_list = []
        self.transform = Transforms()
        self.root_dir = data_root

        # 遍历 XML 数据:root[2] 表示 XML 中的第三个元素，即 <images> 部分，其中包含了每张图像的标注信息
        data_list = []
        with open(file_path, "r") as file_:
            for line in file_.readlines():
                line = line.strip().split(" ")
                coords = np.array([float(xy) for xy in line[:196]]).reshape(-1, 2)
                path_ = line[-1]
                data_path = os.path.join(self.root_dir, path_)
                data_list.append([data_path, coords])

        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # 读取图像以及关键点坐标
        file_path = self.data_list[item][0]
        image = cv2.imread(file_path)  # 以彩色模式读取图像
        landmarks = np.array([float(x) for x in self.wfwl68[item][1:]]).reshape(-1, 2) #self.data_list[item][1] #

        if self.transform:
            # 如果存在预处理变换，应用变换
            image, landmarks = self.transform(image, landmarks)
        # landmarks = landmarks.numpy()
        # image = np.array(image)
        # for pi in range(landmarks.shape[0]):
        #     coord = (landmarks[pi] * cfg.IMG_Width).astype(np.int32)
        #     image = cv2.circle(image, (coord[0], coord[1]), 1, (255, 0, 0) , thickness=2)
        #
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        image = image.numpy()




        return image, landmarks


# 创建数据集对象，并应用预处理变换
#dataset = TrainData(Transforms())
