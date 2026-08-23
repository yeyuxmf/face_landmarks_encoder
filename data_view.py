

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









if __name__ == "__main__":
    tsv_flie = "E:/DataSet/face_data/WFLW/test_blur_metadata.tsv"
    #tsv_flie = "E:/DataSet/face_data/WFLW/test_expression_metadata.tsv"
    #tsv_flie = "E:/DataSet/face_data/WFLW/test_illumination_metadata.tsv"
    #tsv_flie = "E:/DataSet/face_data/WFLW/test_largepose_metadata.tsv"
    #tsv_flie = "E:/DataSet/face_data/WFLW/test_makeup_metadata.tsv"
    # tsv_flie = "E:/DataSet/face_data/WFLW/test_occlusion_metadata.tsv"
    items = pd.read_csv(tsv_flie, sep="\t")
    image_dir = "E:/DataSet/face_data/WFLW/WFLW_images/"
    save_img_root = "./save_img_view/"
    landmark_num = 98
    for index  in range(3, len(items)):
        image_path = items.iloc[index, 0]
        image_path = image_path.replace('\\', '/')
        # wflw testset
        image_path = image_path.replace(
            '//msr-facestore/Workspace/MSRA_EP_Allergan/users/yanghuan/training_data/wflw/rawImages/', '')
        # trainset
        image_path = image_path.replace('./rawImages/', '')
        image_path = os.path.join(image_dir, image_path)
        img = cv2.imread(image_path)
        print(index, "  ", image_path)

        landmarks_target = items.iloc[index, 2]
        landmarks = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            landmark_num, 2)

        H, W, _ = img.shape

        minx, miny = np.min(landmarks[:, 0]), np.min(landmarks[:, 1])
        maxx,maxy = np.max(landmarks[:, 0]), np.max(landmarks[:, 1])
        width = maxx - minx
        height = maxy - miny
        minx, maxx = max(int(minx - width*0.1), 0), min(int(maxx + width*0.1), W)
        miny, maxy= max(int(miny - height*0.1), 0), min(int(maxy + height*0.1), H)
        landmarks = landmarks -np.array([[minx, miny]])

        img = img[miny:maxy, minx: maxx, :]
        for pi in range(landmarks.shape[0]):
            coord = landmarks[pi].astype(np.int32)
            #if pi ==36 or pi ==45:
            img = cv2.circle(img, (coord[0], coord[1]), 1, (255, 0, 0) , thickness=2)
            #img = cv2.putText(img, str(pi), (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),1)
        cv2.imwrite(save_img_root + os.path.basename(image_path), img)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(0)



    print("over")