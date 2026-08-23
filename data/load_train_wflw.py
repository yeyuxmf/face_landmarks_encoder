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
import pandas as pd
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from config import config as cfg
from data.augmentation import Flip
points_index = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]
flip_mapping = (
    [0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
    [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # cheek
    [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
    [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
    [55, 59], [56, 58],
    [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
    [88, 92], [89, 91], [95, 93], [96, 97]
)
def convert_wflw(file_path):
    with open(file_path, 'r') as f:
        annos = f.readlines()
    annos = [x.strip().split() for x in annos]
    annos_new = []
    for anno in annos:
        annos_new.append([])
        # name
        anno = anno[-1:] + anno[:-1]

        annos_new[-1].append(anno[0])
        anno = anno[1:]
        # jaw
        for i in range(17):
            annos_new[-1].append(anno[i*2*2])
            annos_new[-1].append(anno[i*2*2+1])
        # left eyebrow
        annos_new[-1].append(anno[33*2])
        annos_new[-1].append(anno[33*2+1])
        annos_new[-1].append(anno[34*2])
        annos_new[-1].append(str((float(anno[34*2+1])+float(anno[41*2+1]))/2))
        annos_new[-1].append(anno[35*2])
        annos_new[-1].append(str((float(anno[35*2+1])+float(anno[40*2+1]))/2))
        annos_new[-1].append(anno[36*2])
        annos_new[-1].append(str((float(anno[36*2+1])+float(anno[39*2+1]))/2))
        annos_new[-1].append(anno[37*2])
        annos_new[-1].append(str((float(anno[37*2+1])+float(anno[38*2+1]))/2))
        # right eyebrow
        annos_new[-1].append(anno[42*2])
        annos_new[-1].append(str((float(anno[42*2+1])+float(anno[50*2+1]))/2))
        annos_new[-1].append(anno[43*2])
        annos_new[-1].append(str((float(anno[43*2+1])+float(anno[49*2+1]))/2))
        annos_new[-1].append(anno[44*2])
        annos_new[-1].append(str((float(anno[44*2+1])+float(anno[48*2+1]))/2))
        annos_new[-1].append(anno[45*2])
        annos_new[-1].append(str((float(anno[45*2+1])+float(anno[47*2+1]))/2))
        annos_new[-1].append(anno[46*2])
        annos_new[-1].append(anno[46*2+1])
        # nose
        for i in range(51, 60):
            annos_new[-1].append(anno[i*2])
            annos_new[-1].append(anno[i*2+1])
        # left eye
        annos_new[-1].append(anno[60*2])
        annos_new[-1].append(anno[60*2+1])
        annos_new[-1].append(str(0.666*float(anno[61*2])+0.333*float(anno[62*2])))
        annos_new[-1].append(str(0.666*float(anno[61*2+1])+0.333*float(anno[62*2+1])))
        annos_new[-1].append(str(0.666*float(anno[63*2])+0.333*float(anno[62*2])))
        annos_new[-1].append(str(0.666*float(anno[63*2+1])+0.333*float(anno[62*2+1])))
        annos_new[-1].append(anno[64*2])
        annos_new[-1].append(anno[64*2+1])
        annos_new[-1].append(str(0.666*float(anno[65*2])+0.333*float(anno[66*2])))
        annos_new[-1].append(str(0.666*float(anno[65*2+1])+0.333*float(anno[66*2+1])))
        annos_new[-1].append(str(0.666*float(anno[67*2])+0.333*float(anno[66*2])))
        annos_new[-1].append(str(0.666*float(anno[67*2+1])+0.333*float(anno[66*2+1])))
        # right eye
        annos_new[-1].append(anno[68*2])
        annos_new[-1].append(anno[68*2+1])
        annos_new[-1].append(str(0.666*float(anno[69*2])+0.333*float(anno[70*2])))
        annos_new[-1].append(str(0.666*float(anno[69*2+1])+0.333*float(anno[70*2+1])))
        annos_new[-1].append(str(0.666*float(anno[71*2])+0.333*float(anno[70*2])))
        annos_new[-1].append(str(0.666*float(anno[71*2+1])+0.333*float(anno[70*2+1])))
        annos_new[-1].append(anno[72*2])
        annos_new[-1].append(anno[72*2+1])
        annos_new[-1].append(str(0.666*float(anno[73*2])+0.333*float(anno[74*2])))
        annos_new[-1].append(str(0.666*float(anno[73*2+1])+0.333*float(anno[74*2+1])))
        annos_new[-1].append(str(0.666*float(anno[75*2])+0.333*float(anno[74*2])))
        annos_new[-1].append(str(0.666*float(anno[75*2+1])+0.333*float(anno[74*2+1])))
        # mouth
        for i in range(76, 96):
            annos_new[-1].append(anno[i*2])
            annos_new[-1].append(anno[i*2+1])
    return annos_new

class Transforms():
    def __init__(self):
        pass
    def rotate(self, image, landmarks, angle):
        # 随机生成一个在 -angle 到 +angle 范围内的旋转角度
        angle = random.uniform(-angle, +angle)

        # 基于二维平面上的旋转变换的数学特性构建旋转矩阵
        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        # 对图像进行旋转：相比于 PIL 的图像旋转计算开销更小
        imager = imutils.rotate(np.array(image), angle)

        # 将关键点坐标中心化：简化旋转变换的计算，同时确保关键点的变换和图像变换的对应关系
        landmarks_ = landmarks - 0.5
        # 将关键点坐标应用旋转矩阵
        new_landmarks = np.matmul(landmarks_, transformation_matrix)
        # 恢复关键点坐标范围
        new_landmarks = new_landmarks + 0.5
        if torch.sum(new_landmarks<0.0001):
            imager = np.array(image)
            new_landmarks = landmarks
        image = imager

        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        # 调整图像大小
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        # 定义颜色调整的参数：亮度、对比度、饱和度和色调
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        # 对图像进行颜色调整
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks):
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
        sacle = np.random.randint(5, 30, 1)[0] / 100.0
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

    def __call__(self, image, landmarks):
        # 将图像从数组转换为 PIL 图像对象
        image = Image.fromarray(image)
        # 裁剪图像并调整关键点
        image, landmarks = self.crop_face(image, landmarks)
        # 调整图像大小
        image, landmarks = self.resize(image, landmarks, (cfg.IMG_Width, cfg.IMG_Height))
        # 对图像进行颜色调整
        image, landmarks = self.color_jitter(image, landmarks)
        # 对图像和关键点进行旋转变换
        image, landmarks = self.rotate(image, landmarks, angle=10)

        # 将图像从 PIL 图像对象转换为 Torch 张量
        # image = TF.to_tensor(image)
        # # 标准化图像像素值
        # image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks


class TrainData(Dataset):
    def __init__(self, file_path, data_root):
        # 解析 XML 文件

        self._flip = Flip(flip_mapping, 0.5)
        self.items = pd.read_csv(file_path, sep="\t")
        # 初始化变量
        self.data_list = []
        self.transform = Transforms()
        self.root_dir = data_root
        # 遍历 XML 数据:root[2] 表示 XML 中的第三个元素，即 <images> 部分，其中包含了每张图像的标注信息

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # 读取图像以及关键点坐标
        image_path = self.items.iloc[index, 0]
        landmarks_5pts = self.items.iloc[index, 1]
        landmarks_target = self.items.iloc[index, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            self.landmark_num, 2)
        if len(self.items.iloc[index]) > 6:
            tags = np.array(list(map(lambda x: int(float(x)), self.items.iloc[index, 6].split(","))))
        else:
            tags = np.array([])

        image_path = image_path.replace('\\', '/')
        # wflw testset
        image_path = image_path.replace(
            '//msr-facestore/Workspace/MSRA_EP_Allergan/users/yanghuan/training_data/wflw/rawImages/', '')
        # trainset
        image_path = image_path.replace('./rawImages/', '')
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path)
        if self.transform:
            image, landmarks = self._flip.process(image, landmarks_target)
            # 如果存在预处理变换，应用变换
            image, landmarks = self.transform(image, landmarks)
        #
        # # label = landmarks.numpy()*np.array([[cfg.IMG_Width, cfg.IMG_Height]]) /16.0
        hotmap = np.zeros([0, 0, 0])#genarater_hotmap(label, cfg.IMG_Width//16, cfg.IMG_Height//16, sigma=10/16, sizek = 30//16)


        landmarks = landmarks.numpy()
        image = np.array(image)
        for pi in range(landmarks.shape[0]):
            if pi==36 or pi ==45:
                coord = (landmarks[pi] * cfg.IMG_Width).astype(np.int32)
                image = cv2.circle(image, (coord[0], coord[1]), 1, (255, 0, 0) , thickness=2)



        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", image)
        cv2.waitKey(0)

        # hotmap = torch.tensor(hotmap)



        return image, landmarks, hotmap

def genarater_hotmap(label_, IMG_Width, IMG_Height, sigma=10, sizek = 30):

    sigma2 = 1/(sigma*sigma)
    X = np.arange(0, IMG_Width)
    Y = np.arange(0, IMG_Height)
    Y, X = np.meshgrid(Y, X, indexing='ij')
    hotmap = np.zeros((1, IMG_Height, IMG_Width), np.float32)
    for i in range(label_.shape[0]):
        corrd = label_[i]
        u1, u2 = min(int(round(corrd[0])), IMG_Width-1), min(int(round(corrd[1])),IMG_Height-1)
        x1, x2 = max(u1- sizek, 0), min(u1+ sizek+1, IMG_Width-1)
        y1, y2 = max(u2- sizek, 0), min(u2+ sizek+1, IMG_Height-1)
        X_ = X[y1: y2, x1: x2]
        Y_ = Y[y1: y2, x1: x2]

        XU1 = (X_ - u1) * (X_ - u1)
        YU2 = (Y_ - u2) * (Y_ - u2)
        Va = -(XU1+YU2) *sigma2
        gauv = np.exp(Va).astype(np.float32)

        hotmap[0, y1: y2, x1: x2] = np.maximum(hotmap[0,y1: y2, x1: x2], gauv)

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", np.sum(hotmap, axis=0).astype(np.float32))
    # cv2.waitKey(0)

    # hotmap[cfg.PointNms, :, :] = 1-hotmapb

    return hotmap
