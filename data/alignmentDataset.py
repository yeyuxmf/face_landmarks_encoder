import os
import sys
import cv2
import math
import copy
import hashlib
import imageio
import numpy as np
import pandas as pd
from scipy import interpolate
from PIL import Image, ImageEnhance, ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append("./")
from data.augmentation import Augmentation



class AlignmentDataset(Dataset):

    def __init__(self, tsv_flie, image_dir="", transform=None,
                 width=256, height=256, channels=3,
                 means=(127.5, 127.5, 127.5), scale=1 / 127.5,
                 classes_num=None, crop_op=True, aug_prob=0.0, edge_info=None, flip_mapping=None, is_train=True,
                 encoder_type='default',
                 ):
        super(AlignmentDataset, self).__init__()
        self.use_AAM = True
        self.encoder_type = encoder_type

        self.items = pd.read_csv(tsv_flie, sep="\t")
        self.image_dir = image_dir
        self.landmark_num = classes_num[0]
        self.transform = transform

        self.image_width = width
        self.image_height = height
        self.channels = channels
        assert self.image_width == self.image_height

        self.means = means
        self.scale = scale

        self.aug_prob = aug_prob
        self.edge_info = edge_info
        self.is_train = is_train
        std_lmk_5pts = np.array([
            196.0, 226.0,
            316.0, 226.0,
            256.0, 286.0,
            220.0, 360.4,
            292.0, 360.4], np.float32) / 256.0 - 1.0
        std_lmk_5pts = np.reshape(std_lmk_5pts, (5, 2))  # [-1 1]
        target_face_scale = 1.0 if crop_op else 1.25

        self.augmentation = Augmentation(
            is_train=self.is_train,
            aug_prob=self.aug_prob,
            image_size=self.image_width,
            crop_op=crop_op,
            std_lmk_5pts=std_lmk_5pts,
            target_face_scale=target_face_scale,
            flip_rate=0.5,
            flip_mapping=flip_mapping,
            random_shift_sigma=0.05,
            random_rot_sigma=math.pi / 180 * 18,
            random_scale_sigma=0.1,
            random_gray_rate=0.2,
            random_occ_rate=0.4,
            random_blur_rate=0.3,
            random_gamma_rate=0.2,
            random_nose_fusion_rate=0.2)

    def _circle(self, img, pt, sigma=1.0, label_type='Gaussian'):
        # Check that any part of the gaussian is in-bounds
        tmp_size = sigma * 3
        ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
        br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
        if (ul[0] > img.shape[1] - 1 or ul[1] > img.shape[0] - 1 or
                br[0] - 1 < 0 or br[1] - 1 < 0):
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        if label_type == 'Gaussian':
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        else:
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = 255 * g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

    def _polylines(self, img, lmks, is_closed, color=255, thickness=1, draw_mode=cv2.LINE_AA,
                   interpolate_mode=cv2.INTER_AREA, scale=4):
        h, w = img.shape
        img_scale = cv2.resize(img, (w * scale, h * scale), interpolation=interpolate_mode)
        lmks_scale = (lmks * scale + 0.5).astype(np.int32)
        cv2.polylines(img_scale, [lmks_scale], is_closed, color, thickness * scale, draw_mode)
        img = cv2.resize(img_scale, (w, h), interpolation=interpolate_mode)
        return img

    def _generate_edgemap(self, points, scale=0.25, thickness=1):
        h, w = self.image_height, self.image_width
        edgemaps = []
        for is_closed, indices in self.edge_info:
            edgemap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            part = copy.deepcopy(points[np.array(indices)])

            part = self._fit_curve(part, is_closed)
            part[:, 0] = np.clip(part[:, 0], 0, w - 1)
            part[:, 1] = np.clip(part[:, 1], 0, h - 1)
            edgemap = self._polylines(edgemap, part, is_closed, 255, thickness)

            edgemaps.append(edgemap)
        edgemaps = np.stack(edgemaps, axis=0) / 255.0
        edgemaps = torch.from_numpy(edgemaps).float().unsqueeze(0)
        edgemaps = F.interpolate(edgemaps, size=(int(w * scale), int(h * scale)), mode='bilinear',
                                 align_corners=False).squeeze()
        return edgemaps

    def _fit_curve(self, lmks, is_closed=False, density=5):
        try:
            x = lmks[:, 0].copy()
            y = lmks[:, 1].copy()
            if is_closed:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
            tck, u = interpolate.splprep([x, y], s=0, per=is_closed, k=3)
            # bins = (x.shape[0] - 1) * density + 1
            # lmk_x, lmk_y = interpolate.splev(np.linspace(0, 1, bins), f)
            intervals = np.array([])
            for i in range(len(u) - 1):
                intervals = np.concatenate((intervals, np.linspace(u[i], u[i + 1], density, endpoint=False)))
            if not is_closed:
                intervals = np.concatenate((intervals, [u[-1]]))
            lmk_x, lmk_y = interpolate.splev(intervals, tck, der=0)
            # der_x, der_y = interpolate.splev(intervals, tck, der=1)
            curve_lmks = np.stack([lmk_x, lmk_y], axis=-1)
            # curve_ders = np.stack([der_x, der_y], axis=-1)
            # origin_indices = np.arange(0, curve_lmks.shape[0], density)

            return curve_lmks
        except:
            return lmks

    def _image_id(self, image_path):
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, image_path)
        return hashlib.md5(open(image_path, "rb").read()).hexdigest()

    def _load_image(self, image_path):
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, image_path)

        try:
            # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)#HWC, BGR, [0-255]
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # HWC, BGR, [0-255]
            assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
        except:
            try:
                img = imageio.imread(image_path)  # HWC, RGB, [0-255]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # HWC, BGR, [0-255]
                assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
            except:
                try:
                    gifImg = imageio.mimread(image_path)  # BHWC, RGB, [0-255]
                    img = gifImg[0]  # HWC, RGB, [0-255]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # HWC, BGR, [0-255]
                    assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
                except:
                    img = None
        return img

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def _transformPoints2D(self, points, matrix):
        """
        points (nx2), matrix (3x3) -> points (nx2)
        """
        dtype = points.dtype

        # nx3
        points = np.concatenate([points, np.ones_like(points[:, [0]])], axis=1)
        points = points @ np.transpose(matrix)  # nx3
        points = points[:, :2] / points[:, [2, 2]]
        return points.astype(dtype)

    def _transformPerspective(self, image, matrix, target_shape):
        """
        image, matrix3x3 -> transformed_image
        """
        return cv2.warpPerspective(
            image, matrix,
            dsize=(target_shape[1], target_shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=0)

    def _norm_points(self, points, h, w, align_corners=False):
        if align_corners:
            des_points = points / torch.tensor([w - 1, h - 1]).to(points).view(1, 2)
        else:
            des_points = (points) / torch.tensor([w, h]).to(points).view(1, 2)
        des_points = torch.clamp(des_points, 0, 1)
        return des_points

    def _denorm_points(self, points, h, w, align_corners=False):
        if align_corners:
            # [-1, +1] -> [0, SIZE-1]
            des_points = (points + 1) / 2 * torch.tensor([w - 1, h - 1]).to(points).view(1, 1, 2)
        else:
            # [-1, +1] -> [-0.5, SIZE-0.5]
            des_points = ((points + 1) * torch.tensor([w, h]).to(points).view(1, 1, 2) - 1) / 2
        return des_points
    def simulate_sunlight_gradient(self, image, start_point=(0, 0), end_point=(255, 255), intensity=0.5,
                                   sun_color=(255, 255, 100)):
        """
        在图像上模拟渐变的太阳光（黄色调）照射效果。
        :param image: 输入的彩色图像，shape=(H, W, 3)，像素值[0,255]
        :param start_point: 渐变起点 (x, y)
        :param end_point: 渐变终点 (x, y)
        :param intensity: 光照强度，0~1
        :param sun_color: 太阳光颜色 (R, G, B)
        :return: 加了光照的图像
        """
        if np.random.random() < 0.5:
            sun_color = (100, 255, 255)
            if np.random.random() < 0.5:
                sun_color = (50, 100, 255)

        H, W, _ = image.shape
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        # 计算每个像素到起点和终点的距离
        dist_start = np.sqrt((x - start_point[0]) ** 2 + (y - start_point[1]) ** 2)
        dist_end = np.sqrt((x - end_point[0]) ** 2 + (y - end_point[1]) ** 2)
        # 归一化渐变权重
        weight = dist_end / (dist_start + dist_end + 1e-8)
        weight = np.clip(weight, 0, 1)
        weight = (weight * intensity)[..., None]
        # 叠加黄色光照
        sun_color = np.array(sun_color).reshape(1, 1, 3)
        result = image * (1 - weight) + sun_color * weight
        return np.clip(result, 0, 255).astype(np.uint8)

    def simulate_point_light(self, image, intensity=1.0, position=(128, 128), ambient=0.3, decay=1.5):
        """
        模拟点光源照射效果
        :param image: 输入的彩色图像，shape=(256, 256, 3)，像素值[0, 255]
        :param intensity: 光照强度，float，通常在0~2之间
        :param position: 点光源位置，二元组(x, y)
        :param ambient: 环境光比例，float，0~1
        :param decay: 光照随距离衰减的指数，float，通常在1~3之间
        :return: 光照变化后的图像
        """
        h, w, _ = image.shape
        y, x = np.mgrid[0:h, 0:w]
        # 计算每个像素到光源的距离
        dist = np.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2)
        # 归一化距离
        dist = dist / np.max(dist)
        # 计算光照强度
        light = ambient + intensity * (1 - dist) ** decay
        light = np.clip(light, 0, 1)
        # 应用到图像
        result = image * light[..., np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def augment_hsv(self, im_, hgain=0.5, sgain=0.5, vgain=0.5):
        """Applies HSV color-space augmentation to an image with random gains for hue, saturation, and value."""
        im = copy.deepcopy(im_)
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

        return im

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sample = dict()

        image_path = self.items.iloc[index, 0]
        landmarks_5pts = self.items.iloc[index, 1]
        landmarks_5pts = np.array(list(map(float, landmarks_5pts.split(","))), dtype=np.float32).reshape(5, 2)
        landmarks_target = self.items.iloc[index, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            self.landmark_num, 2)
        scale = float(self.items.iloc[index, 3])
        center_w, center_h = float(self.items.iloc[index, 4]), float(self.items.iloc[index, 5])
        if len(self.items.iloc[index]) > 6:
            tags = np.array(list(map(lambda x: int(float(x)), self.items.iloc[index, 6].split(","))))
        else:
            tags = np.array([])

        # image & keypoints alignment
        image_path = image_path.replace('\\', '/')
        # wflw testset
        image_path = image_path.replace(
            '//msr-facestore/Workspace/MSRA_EP_Allergan/users/yanghuan/training_data/wflw/rawImages/', '')
        # trainset
        image_path = image_path.replace('./rawImages/', '')
        image_path = os.path.join(self.image_dir, image_path)

        # image path
        sample["image_path"] = image_path

        img = self._load_image(image_path)  # HWC, BGR, [0, 255]
        assert img is not None

        # augmentation
        # landmarks_target = [-0.5, edge-0.5]
        img, landmarks_target, matrix = \
            self.augmentation.process(img, landmarks_target, landmarks_5pts, scale, center_w, center_h)


        # if self.is_train: #huang
        #     signxv = 2 * np.random.rand(1)[0] - 1
        #     signyv = 2 * np.random.rand(1)[0] - 1
        #     if 0 == np.random.randint(0, 3, 1)[0]:
        #         np.random.random()
        #         landmarks_target = landmarks_target + np.load("variances.npy") * np.array([[signxv, signyv]]) / 4


        landmarks = self._norm_points(torch.from_numpy(landmarks_target), self.image_height, self.image_width)

        sample["label"] = [landmarks, ]
        edgemap = cv2.Sobel(img, cv2.CV_8U, 1, 1)
        edgemap = (edgemap - np.min(edgemap))/(np.max(edgemap)- np.min(edgemap)).astype(np.float32)
        sample["edgemap"] = torch.tensor(edgemap).permute(2, 0, 1)
        # if self.use_AAM:
        # pointmap = self.encoder.generate_heatmap(landmarks_target)
        #     edgemap = self._generate_edgemap(landmarks_target)
        #     sample["label"] += [pointmap, edgemap]
        hotmap = genarater_hotmap(landmarks.numpy(), 32, 32)
        pointmap = hotmap
        # sample['matrix'] = matrix
        # landmarks = landmarks.numpy()*256
        # for pi in range(landmarks.shape[0]):
        #     coord = landmarks[pi].astype(np.int32)
        #     if pi ==36 or pi ==45:
        #         img = cv2.circle(img, (coord[0], coord[1]), 1, (255, 0, 0) , thickness=2)
        #         img = cv2.putText(img, str(pi), (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),1)
        #
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        # if np.random.random() < 0.3:
        #     flag = np.random.randint(0, 3, 1)[0]
        #     if 0==flag:
        #         img = self.augment_hsv(img)
        #     elif 1==flag:
        #         img = self.simulate_point_light(img)
        #     else:
        #         img = self.simulate_sunlight_gradient(img)

        sample["rowimg"] = img
        # image normalization
        img = img.transpose(2, 0, 1).astype(np.float32)  # CHW, BGR, [0, 255]
        img[0, :, :] = (img[0, :, :] - self.means[0]) * self.scale
        img[1, :, :] = (img[1, :, :] - self.means[1]) * self.scale
        img[2, :, :] = (img[2, :, :] - self.means[2]) * self.scale
        sample["data"] = torch.from_numpy(img)  # CHW, BGR, [-1, 1]

        sample["tags"] = tags
        sample["pointmap"] = pointmap
        return sample
def genarater_hotmap(landmarks, IMG_Width, IMG_Height):

    hotmap = np.zeros((landmarks.shape[0], IMG_Height, IMG_Width), np.float32)
    coords = np.round(landmarks * np.array([[IMG_Width, IMG_Height]])).clip(0, IMG_Width-1)
    for i in  range(coords.shape[0]):
        x, y = coords[i]
        x, y = int(round(x)), int(round(y))
        hotmap[i, y, x] = 1
    return torch.tensor(hotmap)