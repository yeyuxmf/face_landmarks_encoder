import os
import math
import numpy as np

import xml.etree.ElementTree as ET
def main(root_dir, fw_path_train, fw_path_test):
    train_lines = []
    test_lines = []
    test_dirs = ['lfpw/testset','ibug', 'helen/testset']
    train_dirs = ['lfpw/trainset', 'afw', 'helen/trainset']
    tree = ET.parse("G:/ibug_300W_large_face_landmark_dataset/labels_ibug_300W.xml")
    root = tree.getroot()

    root_dir = "G:/ibug_300W_large_face_landmark_dataset/"
    # 遍历 XML 数据:root[2] 表示 XML 中的第三个元素，即 <images> 部分，其中包含了每张图像的标注信息

    file_train = open(fw_path_train, 'w')
    file_test = open(fw_path_test, 'w')
    file_commontest = open('G:/ibug_300W_large_face_landmark_dataset/list_commontest.txt', 'w')
    file_challengetest = open('G:/ibug_300W_large_face_landmark_dataset/list_challengetest.txt', 'w')
    for filename in root[2]:
        image_filenames= filename.attrib['file']

        crops= filename[0].attrib

        landmark = image_filenames
        for num in range(68):
            landmark = landmark +","
            x_coordinate = str(filename[0][num].attrib['x'])
            y_coordinate = str(filename[0][num].attrib['y'])
            landmark = landmark + x_coordinate+"," +y_coordinate
        if "testset" in image_filenames or "ibug" in image_filenames:
            test_lines.append(landmark)
            file_test.write(landmark + "\n")
            if "ibug" in image_filenames:
                file_commontest.write(landmark + "\n")
            else:
                file_challengetest.write(landmark + "\n")
        else:
            train_lines.append(landmark)
            file_train.write(landmark + "\n")
    file_train.close()
    file_test.close()
    file_commontest.close()
    file_challengetest.close()

if __name__ == '__main__':
    root_dir = "G:/300w/"
    print(root_dir)
    fw_path_train = os.path.join(root_dir, 'G:/ibug_300W_large_face_landmark_dataset/list_train.txt')
    fw_path_test = os.path.join(root_dir, 'G:/ibug_300W_large_face_landmark_dataset/list_test.txt')
    main(root_dir, fw_path_train, fw_path_test)