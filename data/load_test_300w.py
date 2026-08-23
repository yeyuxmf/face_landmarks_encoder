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

def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(path, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)

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

    def data_flip(self, img, label_coords):
        height, width, _ = img.shape

        flag = np.random.randint(0, 2, 1)[0]
        if 1 == flag:
            img = img[:, ::-1, :]
            label_coords[:, 0] = 1 - label_coords[:, 0]

        return img, label_coords


    def __call__(self, image, landmarks):
        # 将图像从数组转换为 PIL 图像对象
        image = Image.fromarray(image)
        # 裁剪图像并调整关键点
        image, landmarks = self.crop_face(image, landmarks)

        # image = np.ascontiguousarray(image)
        # image, landmarks = self.data_flip(image, landmarks)
        # image = Image.fromarray(image)

        # 调整图像大小
        image, landmarks = self.resize(image, landmarks, (cfg.IMG_Width, cfg.IMG_Height))
        # 对图像进行颜色调整
        # image, landmarks = self.color_jitter(image, landmarks)
        # # 对图像和关键点进行旋转变换
        # image, landmarks = self.rotate(image, landmarks, angle=10)



        # 将图像从 PIL 图像对象转换为 Torch 张量
        image = TF.to_tensor(image)
        # 标准化图像像素值
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks





class TestData(Dataset):
    def __init__(self, file_root, data_root):
        # 解析 XML 文件
        # tree = ET.parse(file_path)
        # root = tree.getroot()
        #, 'ibug'

        file_data = file_root + "list_test.txt"
        # 初始化变量
        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = Transforms()
        self.root_dir = data_root


        with open(file_data, "r") as file_:
            for line in file_.readlines():
                line = line.strip().split(",")
                image_file = self.root_dir + line[0]
                self.image_filenames.append(image_file)
                landmark = [float(line[i]) for i in range(1, len(line), 1)]
                landmark = np.array(landmark).reshape(-1, 2)
                self.landmarks.append(landmark)

        #
        # # 遍历 XML 数据:root[2] 表示 XML 中的第三个元素，即 <images> 部分，其中包含了每张图像的标注信息
        # for filename in root[2]:
        #     self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
        #
        #     self.crops.append(filename[0].attrib)
        #
        #     landmark = []
        #     for num in range(68):
        #         x_coordinate = int(filename[0][num].attrib['x'])
        #         y_coordinate = int(filename[0][num].attrib['y'])
        #         landmark.append([x_coordinate, y_coordinate])
        #     self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')
        self.image_filenames = self.image_filenames

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # 读取图像以及关键点坐标
        image = cv2.imread(self.image_filenames[index])  # 以彩色模式读取图像
        # image = cv2.imread(self.image_filenames[index], 0) # 以灰色模式读取图像
        landmarks = self.landmarks[index].copy()

        if self.transform:
            # 如果存在预处理变换，应用变换
            image, landmarks = self.transform(image, landmarks)

        return image, landmarks


