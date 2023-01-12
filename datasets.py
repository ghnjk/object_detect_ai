#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: datasets
@author: jkguo
@create: 2023/1/2
"""
import os
import requests
from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
