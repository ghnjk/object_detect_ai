#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: image_augmentation
@author: jkguo
@create: 2023/1/14
"""
import random
from PIL import Image, ImageEnhance
import numpy as np


def augmentation_one_picture(image: np.ndarray, label):
    """
    对图像进行增广
    :param image:
    :param label:
    :return:
    """
    aug_images = []
    aug_labels = []
    height = image.shape[0]
    width = image.shape[1]
    image = Image.fromarray((image * 256).astype('uint8'))
    for _ in range(4):
        # 旋转，镜像增广
        image = image.rotate(90)
        x0, y0, x1, y1 = label
        label = (
            y0,
            width - x1,
            y1,
            width - x0
        )
        height, width = width, height
        aug_images.append(image)
        aug_labels.append(label)

        x0, y0, x1, y1 = label
        flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        flip_label = (
            width - x1,
            y0,
            width - x0,
            y1
        )
        aug_images.append(flip_image)
        aug_labels.append(flip_label)
    # 颜色相关增广
    color_aug_images = []
    color_aug_labels = []
    # 亮度增广
    brights = [
        random.uniform(0.2, 0.3),
        random.uniform(0.3, 0.6),
        random.uniform(0.6, 0.9),
        random.uniform(1.5, 3)
    ]
    select_idx_list = random.sample(list(range(len(aug_labels))), len(brights))
    for i in range(len(brights)):
        bright = brights[i]
        image = aug_images[select_idx_list[i]]
        label = aug_labels[select_idx_list[i]]
        en_image = ImageEnhance.Brightness(image).enhance(bright)
        color_aug_images.append(en_image)
        color_aug_labels.append(label)
    # 颜色饱和度增广
    colors = [
        random.uniform(0.1, 0.3),
        random.uniform(0.3, 0.6),
        random.uniform(0.6, 0.9),
        random.uniform(1.5, 3)
    ]
    select_idx_list = random.sample(list(range(len(aug_labels))), len(colors))
    for i in range(len(brights)):
        c = colors[i]
        image = aug_images[select_idx_list[i]]
        label = aug_labels[select_idx_list[i]]
        en_image = ImageEnhance.Color(image).enhance(c)
        color_aug_images.append(en_image)
        color_aug_labels.append(label)
    # 对比度增广
    cons_list = [
        random.uniform(0.1, 0.3),
        random.uniform(0.3, 0.6),
        random.uniform(0.6, 0.9),
        random.uniform(1.5, 3)
    ]
    select_idx_list = random.sample(list(range(len(aug_labels))), len(cons_list))
    for i in range(len(brights)):
        c = cons_list[i]
        image = aug_images[select_idx_list[i]]
        label = aug_labels[select_idx_list[i]]
        en_image = ImageEnhance.Contrast(image).enhance(c)
        color_aug_images.append(en_image)
        color_aug_labels.append(label)
    # 组装所有样例
    res_images = []
    res_labels = []
    for i in range(len(aug_images)):
        res_images.append(
            np.array(aug_images[i]) / 256.0
        )
        res_labels.append(
            aug_labels[i]
        )
    for i in range(len(color_aug_images)):
        res_images.append(
            np.array(color_aug_images[i]) / 256.0
        )
        res_labels.append(
            color_aug_labels[i]
        )
    return np.array(res_images), np.array(res_labels)


def augmentation_samples(images, labels):
    """
    对图像进行增广所有的样本
    :param images:
    :param labels:
    :return:
    """
    aug_images = []
    aug_labels = []
    for i in range(len(images)):
        item_images, item_labels = augmentation_one_picture(
            images[i], labels[i]
        )
        aug_images.extend(item_images)
        aug_labels.extend(item_labels)
    return np.array(aug_images), np.array(aug_labels)
