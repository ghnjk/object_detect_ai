#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: voc_seg_classify
@author: jkguo
@create: 2023/1/28
"""
import os
import random
import typing
from fit_common.utils import file_util
import numpy as np
import tensorflow as tf

try:
    import keras
    from keras.layers import Convolution2D, Conv2DTranspose
    from keras.utils import Sequence
    from keras.activations import softmax
    from keras.optimizers import Adam
except ImportError:
    from tensorflow import keras
    from tensorflow.keras.layers import Convolution2D, Conv2DTranspose
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.activations import softmax
    from tensorflow.keras.optimizers import Adam
from datasets import VocDatasets


def bi_linear_kernel_initializer(shape, dtype=None):
    """
    双线性插值（bilinear interpolation）
    https://zh.d2l.ai/chapter_computer-vision/fcn.html
    :param shape:
    :param dtype:
    :return:
    """
    in_channels = shape[2]
    out_channels = shape[3]
    kernel_size = shape[0]
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    fill_core = (1 - tf.abs(og[0] - center) / factor) * \
                (1 - tf.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels,
                       kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = fill_core
    weight = np.transpose(weight, (2, 3, 0, 1))
    print(f"bi_linear_kernel_initializer return shape : {weight.shape}")
    return weight


class VocSegClassifyModel(object):

    def __init__(self, class_count: int, image_row_count: int = 160, image_col_count: int = 240,
                 image_channel_count: int = 3,
                 channel_format: str = "channels_last"):
        self.class_count = class_count
        self.image_row_count = image_row_count
        self.image_col_count = image_col_count
        self.image_channel_count = image_channel_count
        self.channel_format = channel_format
        self.inputs = None
        self.outputs = None
        self.model = None

    def build_net(self):
        with tf.name_scope("inputs"):
            self.inputs = keras.Input(shape=(self.image_row_count, self.image_col_count, self.image_channel_count),
                                      name="input_image")
        x = self.inputs
        # net = tf.keras.applications.ResNet50V2(weights="imagenet",
        #                                        input_shape=(self.image_row_count, self.image_col_count,
        #                                                     self.image_channel_count),
        #                                        include_top=False)
        # # net.trainable = False
        # x = net(x)
        net = tf.keras.applications.VGG16(weights="imagenet",
                                          input_shape=(self.image_row_count, self.image_col_count,
                                                       self.image_channel_count),
                                          include_top=False)
        for layer in net.layers[: -4]:
            if layer.name.startswith("input"):
                continue
            # print(f"layer {layer.name} shape {layer.output_shape}")
            # layer.trainable = False
            x = layer(x)
        x = Convolution2D(filters=self.class_count, kernel_size=(1, 1),
                          data_format=self.channel_format, name="last_1x1_conv")(x)
        x = Conv2DTranspose(filters=self.class_count, kernel_size=32, strides=16, padding="same",
                            # kernel_initializer=bi_linear_kernel_initializer,
                            data_format=self.channel_format, name="transpose_conv")(x)
        x = softmax(x)
        self.outputs = x
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name="VocSegClassify")
        # adam = Adam(learning_rate=1e-3)
        self.model.compile(
            optimizer="rmsprop",
            loss=["sparse_categorical_crossentropy"]
        )
        return self.model

    def save_model(self, model_path: str = "./data/voc_seg_classify.h5"):
        file_util.prepare_parent_folder(model_path)
        self.model.save(model_path)

    def load_model(self, model_path: str = "./data/voc_seg_classify.h5"):
        self.model = keras.models.load_model(model_path, custom_objects={
            "bi_linear_kernel_initializer": bi_linear_kernel_initializer
        })

    def predict(self, images):
        labels = np.argmax(self.model.predict(images), axis=3)
        return labels


class VocDataGenerator(Sequence):

    def __init__(self, batch_size, ds, force_rebuild=False):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.ds = ds
        self.crop_height = self.ds.crop_size[0]
        self.crop_width = self.ds.crop_size[1]
        self.output_height = self.crop_height // 2
        self.output_width = self.crop_width // 2
        self.batch_size: int = batch_size
        img_file = "./data/voc_temp_ds.img.npy"
        label_file = "./data/voc_temp_ds.label.npy"
        if not force_rebuild and os.path.isfile(img_file) and os.path.isfile(label_file):
            self.images = np.load(img_file)
            self.labels = np.load(label_file)
        else:
            images, labels, seg_images = self.ds.load_train_dataset()
            self.images: np.ndarray = images
            self.labels: np.ndarray = labels
            self.__pre_process()
            np.save(img_file, self.images)
            np.save(label_file, self.labels)

    @staticmethod
    def convert_label_to_outputs(labels, crop_size):
        crop_height, crop_width = crop_size
        crop_labels = []
        for idx in range(len(labels)):
            label = np.array(labels[idx])
            offset_r = random.randint(0, label.shape[0] - crop_height)
            offset_c = random.randint(0, label.shape[1] - crop_width)
            offset_re = offset_r + crop_height
            offset_ce = offset_c + crop_width
            label = label[offset_r: offset_re, offset_c: offset_ce]
            # 压缩图片和label
            label = VocDataGenerator.__down_sample_label(label, crop_height, crop_width)
            crop_labels.append(label)
        return np.array(crop_labels)

    @staticmethod
    def convert_image_to_inputs(images, crop_size, normalize_image=True):
        from PIL import Image
        crop_height, crop_width = crop_size
        crop_images = []
        for idx in range(len(images)):
            img = images[idx]
            offset_r = random.randint(0, img.shape[0] - crop_height)
            offset_c = random.randint(0, img.shape[1] - crop_width)
            offset_re = offset_r + crop_height
            offset_ce = offset_c + crop_width
            img = img[offset_r: offset_re, offset_c: offset_ce]
            img = np.asarray(
                Image.fromarray(img).resize(
                    (crop_width // 2, crop_height // 2)
                )
            )
            if normalize_image:
                img = VocDataGenerator.normalize_image(img)
            crop_images.append(img)
        return np.array(crop_images)

    def __pre_process(self):
        print("pre process data...")
        crop_size = (self.crop_height, self.crop_width)
        self.images = self.convert_image_to_inputs(self.images, crop_size, normalize_image=False)
        self.labels = self.convert_label_to_outputs(self.labels, crop_size)
        print("pre process done.")

    @staticmethod
    def normalize_image(image):
        return np.array(image, dtype=np.float32) / 255.0
        # return (np.array(image, dtype=np.float32) / 255.0 - self.rgb_mean) / self.rgb_std

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        result = self.normalize_image(self.images[start: end]), self.labels[start: end]
        # self.ds.convert_label_to_probes(
        #     self.labels[start: end]
        # )
        # print(f"get item {idx} at [{start}: {end})")
        return result

    @staticmethod
    def __down_sample_label(label, image_height, image_width):
        label = np.array(label)
        new_label = []
        for i in range(image_height // 2):
            new_row = []
            for j in range(image_width // 2):
                a = label[2 * i: 2 * i + 1, 2 * j: 2 * j + 1].flatten()
                counts = np.bincount(a)
                new_row.append(np.argmax(counts))
            new_label.append(new_row)
        return np.array(new_label)


class VocSegTrainer(object):

    def __init__(self, voc_seg_model: VocSegClassifyModel = None):
        self.ds = VocDatasets()
        self.train_gen: typing.Optional[VocDataGenerator] = None
        self.test_gen: typing.Optional[VocDataGenerator] = None
        self.image_height = self.ds.crop_size[0]
        self.image_width = self.ds.crop_size[1]
        self.voc_seg_model: typing.Optional[VocSegClassifyModel] = voc_seg_model

    def prepare_datasets(self, batch_size):
        print("loading train data...")
        self.train_gen = VocDataGenerator(batch_size, self.ds)
        # print("loading test data...")
        # self.test_images, self.test_labels, self.test_label_probs, self.test_seg_images = self.ds.load_test_dataset()
        print("prepare_datasets done.")

    def train(self, epoch=20, batch_size=32, load_model=False):
        if self.train_gen is None:
            self.prepare_datasets(batch_size=10000)
        if self.voc_seg_model is None:
            self.voc_seg_model = VocSegClassifyModel(self.ds.class_count(),
                                                     image_row_count=self.train_gen.output_height,
                                                     image_col_count=self.train_gen.output_width)
            if not load_model:
                self.voc_seg_model.build_net()
        if load_model:
            self.voc_seg_model.load_model()
        log_dir = "./log"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        inputs, outputs = self.train_gen[0]
        self.voc_seg_model.model.fit(inputs, outputs, batch_size=batch_size, epochs=epoch,
                                     callbacks=[tensorboard_callback])
        # all_len = len(self.train_gen)
        # for e in range(epoch):
        #     for i in range(all_len):
        #         inputs, outputs = self.train_gen[i]
        #         self.voc_seg_model.model.fit(inputs, outputs, batch_size=batch_size, epochs=1,
        #                                      callbacks=[tensorboard_callback])
