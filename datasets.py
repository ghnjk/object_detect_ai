#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: datasets
@author: jkguo
@create: 2023/1/2
"""

import os
import tarfile
from zipfile import ZipFile

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

D2L_DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATA_DIR = "./data"


class D2LBananaDatasets(object):

    def __init__(self, data_dir: str = DATA_DIR):
        self.banana_dataset_zip_filename: str = "banana-detection.zip"
        self.banana_detect_uri: str = os.path.join(D2L_DATA_URL, self.banana_dataset_zip_filename)
        self.banana_detect_zip_file: str = os.path.join(data_dir, os.path.basename(self.banana_detect_uri))
        self.banana_data_dir: str = os.path.join(data_dir,
                                                 os.path.splitext(os.path.basename(self.banana_dataset_zip_filename))[0]
                                                 )
        self.train_data_dir: str = os.path.join(self.banana_data_dir, "bananas_train")
        self.test_data_dir: str = os.path.join(self.banana_data_dir, "bananas_val")

    def load_train_dataset(self):
        self.__prepare_dataset()
        train_labels = self.__load_label(os.path.join(self.train_data_dir, "label.csv"))
        train_images = self.__load_images(os.path.join(self.train_data_dir, "images"), len(train_labels))
        return train_images, train_labels

    def load_test_dataset(self):
        self.__prepare_dataset()
        test_labels = self.__load_label(os.path.join(self.test_data_dir, "label.csv"))
        test_images = self.__load_images(os.path.join(self.test_data_dir, "images"), len(test_labels))
        return test_images, test_labels

    def __prepare_dataset(self):
        if os.path.isdir(self.train_data_dir) and os.path.isdir(self.test_data_dir):
            return
        # download from web
        if not os.path.isfile(self.banana_detect_zip_file):
            print(f"downloading dataset {self.banana_detect_uri}")
            response = requests.get(self.banana_detect_uri)
            open(self.banana_detect_zip_file, "wb").write(response.content)
            print(f"save to {self.banana_detect_zip_file} success.")
        # extract zip file
        with ZipFile(self.banana_detect_zip_file, 'r') as f:
            f.extractall(os.path.dirname(self.banana_data_dir))
        print(f"extract {self.banana_detect_zip_file} to {self.banana_data_dir}.")

    @staticmethod
    def __load_label(label_csv_file: str):
        df = pd.read_csv(label_csv_file)
        return df[["xmin", "ymin", "xmax", "ymax"]].to_numpy()

    @staticmethod
    def __load_images(image_file_dir: str, image_count):
        image_list = []
        for i in range(image_count):
            file = os.path.join(image_file_dir, f"{i}.png")
            image_list.append(plt.imread(file))
        return image_list

    @staticmethod
    def draw_image(img, label):
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        rect = patches.Rectangle((label[0], label[1]), label[2] - label[0], label[3] - label[1], linewidth=1,
                                 edgecolor='w', facecolor='none')
        plt.gca().add_patch(rect)


class VocDatasets(object):
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    def __init__(self, data_dir: str = DATA_DIR, crop_size=(320, 480)):
        self.crop_size = crop_size
        self.voc_tar_file_name = "VOCtrainval_11-May-2012.tar"
        self.voc_ds_url: str = os.path.join(D2L_DATA_URL, self.voc_tar_file_name)
        self.local_voc_ds_tar_file: str = os.path.join(data_dir, os.path.basename(self.voc_ds_url))
        self.voc_data_dir: str = os.path.join(data_dir, "VOCdevkit/VOC2012")
        self.train_data_sets_file: str = os.path.join(self.voc_data_dir, "ImageSets/Segmentation/trainval.txt")
        self.test_data_sets_file: str = os.path.join(self.voc_data_dir, "ImageSets/Segmentation/val.txt")
        self.image_data_dir: str = os.path.join(self.voc_data_dir, "JPEGImages")
        self.seg_class_image_dir: str = os.path.join(self.voc_data_dir, "SegmentationClass")
        self.voc_color_map: np.ndarray = self.__build_voc_color_map()
        self.voc_color_prob_map: np.ndarray = self.__build_voc_prob_map()

    @staticmethod
    def class_count():
        return len(VocDatasets.VOC_CLASSES)

    def load_train_dataset(self):
        self.__prepare_dataset()
        train_image, train_label, train_seg_img = self.__load_voc_data_sets(
            self.train_data_sets_file
        )
        return train_image, train_label, train_seg_img

    def load_test_dataset(self):
        self.__prepare_dataset()
        test_image, test_label, test_seg_img = self.__load_voc_data_sets(
            self.test_data_sets_file
        )
        return test_image, test_label, test_seg_img

    @staticmethod
    def convert_label_to_outline(label: np.ndarray):
        height = label.shape[0]
        width = label.shape[1]
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1)
        ]
        outline = np.zeros(label.shape, dtype=int)
        for i in range(height):
            for j in range(width):
                all_same = True
                for d in offsets:
                    i1 = i + d[0]
                    j1 = j + d[1]
                    if i1 < 0 or i1 >= height or j1 < 0 or j1 >= width:
                        continue
                    if label[i][j] != label[i1][j1]:
                        all_same = False
                        break
                if not all_same:
                    outline[i][j] = 1
        return outline

    def __prepare_dataset(self):
        if os.path.isfile(self.train_data_sets_file) and os.path.isfile(self.test_data_sets_file) and os.path.isdir(
                self.image_data_dir) and os.path.isdir(self.seg_class_image_dir):
            return
        # download from web
        if not os.path.isfile(self.local_voc_ds_tar_file):
            print(f"downloading dataset {self.local_voc_ds_tar_file}")
            response = requests.get(self.voc_ds_url)
            open(self.local_voc_ds_tar_file, "wb").write(response.content)
            print(f"save to {self.local_voc_ds_tar_file} success.")
        # extract zip file
        with tarfile.open(self.local_voc_ds_tar_file) as fp:
            fp.extractall(os.path.dirname(self.voc_data_dir))
        print(f"extract {self.local_voc_ds_tar_file} to {self.voc_data_dir}.")

    def __load_voc_data_sets(self, data_set_files: str):
        images = []
        labels = []
        seg_images = []
        with open(data_set_files, "r") as fp:
            img_file_name_list = fp.read().split()
        for fn in img_file_name_list:
            jpeg_file_path = os.path.join(self.image_data_dir,
                                          f"{fn}.jpg"
                                          )
            img = plt.imread(jpeg_file_path)
            seg_img_file_path = os.path.join(self.seg_class_image_dir,
                                             f"{fn}.png"
                                             )
            seg_img = plt.imread(seg_img_file_path)
            seg_img = np.array(seg_img[:, :, :3] * 255, dtype="uint8")
            if img.shape[0] > img.shape[1]:
                img = np.transpose(img, (1, 0, 2))
                seg_img = np.transpose(seg_img, (1, 0, 2))
                # print(len(images))
            if img.shape[0] < self.crop_size[0] or img.shape[1] < self.crop_size[1]:
                continue
            images.append(img)
            seg_images.append(seg_img)
            label = self.__convert_seg_to_label(seg_img)
            labels.append(label)
        return images, labels, seg_images

    def __convert_seg_to_label(self, seg_img):
        seg_img = seg_img.astype(int)
        idx = (seg_img[:, :, 0] * 256 + seg_img[:, :, 1]) * 256 + seg_img[:, :, 2]
        labels = self.voc_color_map[idx]
        return labels

    def convert_label_to_probes(self, labels):
        # invalid_idx = [i for i, v in enumerate(labels.reshape((-1, 1))) if v < 0]
        # if len(invalid_idx) > 0:
        #     print(f"has invalid labels: ")
        #     for idx in invalid_idx:
        #         print(seg_img[int(idx / seg_img.shape[1]), idx % seg_img.shape[1], :])
        #     exit(0)
        return self.voc_color_prob_map[labels]

    @staticmethod
    def convert_label_to_seg_image(labels):
        mp = np.array(VocDatasets.VOC_COLORMAP, dtype="uint8")
        return mp[labels]

    @staticmethod
    def __build_voc_color_map():
        mp = np.zeros(256 ** 3, dtype=int) - 1
        for i, item in enumerate(VocDatasets.VOC_COLORMAP):
            idx = (item[0] * 256 + item[1]) * 256 + item[2]
            mp[idx] = i
        outline_color = [224, 224, 192]
        idx = (outline_color[0] * 256 + outline_color[1]) * 256 + outline_color[2]
        mp[idx] = 0
        return mp

    @staticmethod
    def __build_voc_prob_map():
        cls_count = len(VocDatasets.VOC_COLORMAP)
        mp = np.zeros((cls_count, cls_count), dtype=np.float32)
        for i in range(cls_count):
            mp[i][i] = 1.0
        # for i, item in enumerate(VocDatasets.VOC_COLORMAP):
        #     idx = (item[0] * 256 + item[1]) * 256 + item[2]
        #     mp[idx][i] = 1
        # outline_color = [224, 224, 192]
        # idx = (outline_color[0] * 256 + outline_color[1]) * 256 + outline_color[2]
        # mp[idx][0] = 1
        return mp
