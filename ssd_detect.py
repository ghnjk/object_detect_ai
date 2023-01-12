#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: ssd_detect
@author: jkguo
@create: 2023/1/8
"""
import tensorflow as tf
import keras
import numpy as np
from anchor_box import gen_all_anchors, to_box, DetectAnchorBoxMatcher
from fit_common.utils import file_util
from keras.layers import Convolution2D, Activation, MaxPooling2D, BatchNormalization, GlobalMaxPool2D
from keras.activations import softmax
import typing

LAYER_PARAMS = [
    {
        "h": 32,
        "w": 32,
        "scales": [0.1, 0.2],
        "aspect_ratios": [1, 0.5, 2]
    },
    {
        "h": 16,
        "w": 16,
        "scales": [0.3, 0.4],
        "aspect_ratios": [1, 0.5, 2]
    },
    {
        "h": 8,
        "w": 8,
        "scales": [0.5, 0.6],
        "aspect_ratios": [1, 0.5, 2]
    },
    {
        "h": 4,
        "w": 4,
        "scales": [0.7, 0.8],
        "aspect_ratios": [1, 0.5, 2]
    },
    {
        "h": 1,
        "w": 1,
        "scales": [0.85, 0.95],
        "aspect_ratios": [1, 0.5, 2]
    },
]


def tiny_ssd_loss(y_true, y_pred):
    """
    计算loss
    :param y_true:
        shape [batch_size,all layer( row * col * anchor_count), class+1], 锚框分类
                    [batch_size,all layer( row * col * anchor_count), 4], 锚框偏移量
                    [batch_size,all layer( row * col * anchor_count), 4] 锚框mask
    :param y_pred:
        shape [batch_size,all layer( row * col * anchor_count), class+1],
                    [batch_size,all layer( row * col * anchor_count), 4]
                    // ignored
    :return: loss
    """
    cce = keras.losses.CategoricalCrossentropy(name="class_cce")
    mae = keras.losses.MeanAbsoluteError(name="offset_mae")
    return tf.add(
        cce(y_true[0], y_pred[0]),
        mae(y_true[1], y_pred[1]),
        name="tiny_ssd_loss"
    )
    # return cce(y_true[0], y_pred[0]) + mae(y_true[1] * y_true[2], y_pred[1] * y_true[2])


class TinySSD(object):

    def __init__(self, image_row_count: int = 256, image_col_count: int = 256, image_channel_count: int = 3,
                 channel_format: str = "channels_last", anchor_count_per_unit: int = 4, class_count: int = 1):
        self.image_row_count: int = image_row_count
        self.image_col_count: int = image_col_count
        self.image_channel_count: int = image_channel_count
        self.channel_format: str = channel_format
        self.anchor_count_per_unit: int = anchor_count_per_unit
        self.class_count: int = class_count
        if self.channel_format == "channels_last":
            self.batch_normal_axis = 3
        else:
            self.batch_normal_axis = 1
        self.anchor_pred_layers = []
        self.all_anchor_count: int = 0
        self.inputs = None
        self.outputs = None
        self.model: typing.Optional[keras.Model] = None
        self.all_anchors = gen_all_anchors(self.image_row_count, self.image_col_count, LAYER_PARAMS)

    def build_net(self):
        # 256, 256, 3
        self.inputs = keras.Input(shape=(self.image_row_count, self.image_col_count, self.image_channel_count),
                                  name="input_image")
        vgg16 = tf.keras.applications.VGG16(weights="imagenet",
                                            input_shape=(self.image_row_count, self.image_col_count,
                                                         self.image_channel_count),
                                            include_top=False)
        x = self.inputs
        for layer in vgg16.layers:
            if layer.name.startswith("input"):
                continue
            layer.trainable = False
            x = layer(x)
            if layer.output_shape[1] <= 64:
                break
        with tf.name_scope("image_pre_procsess"):
            x = self.add_down_sample_blk(x, 64)  # 32, 32, 64
        with tf.name_scope("layer_1_anchors"):
            self.anchor_pred_layers.append(
                self.add_anchor_pred_layer(x)
            )
            x = self.add_down_sample_blk(x, 128)  # 16, 16, 128
        with tf.name_scope("layer_2_anchors"):
            self.anchor_pred_layers.append(
                self.add_anchor_pred_layer(x)
            )
        x = self.add_down_sample_blk(x, 128)  # 8, 8, 128
        with tf.name_scope("layer_3_anchors"):
            self.anchor_pred_layers.append(
                self.add_anchor_pred_layer(x)
            )
        x = self.add_down_sample_blk(x, 128)  # 4, 4, 128
        with tf.name_scope("layer_4_anchors"):
            self.anchor_pred_layers.append(
                self.add_anchor_pred_layer(x)
            )
        x = GlobalMaxPool2D(data_format=self.channel_format, keepdims=True)(x)
        with tf.name_scope("layer_5_anchors"):
            self.anchor_pred_layers.append(
                self.add_anchor_pred_layer(x)
            )
        self.outputs = self.combine_all_pred_layers()
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name="TinySSD")
        adam = keras.optimizers.Adam(learning_rate=1e-2)
        self.model.compile(
            optimizer=adam,
            loss=["categorical_crossentropy", "mae"])
        return self.model

    def add_down_sample_blk(self, inputs, filter_count: int):
        """
        添加2层cnn+一次max pool。row和col会减半
        :param inputs: input tensor
        :param filter_count: output filter count
        :return: output tensor
        """
        x = Convolution2D(filters=filter_count, kernel_size=(3, 3),
                          padding="same",
                          data_format=self.channel_format
                          )(inputs)
        x = BatchNormalization(axis=self.batch_normal_axis)(x)
        x = Activation('relu')(x)
        x = Convolution2D(filters=filter_count, kernel_size=(3, 3),
                          padding="same",
                          data_format=self.channel_format
                          )(x)
        x = BatchNormalization(axis=self.batch_normal_axis)(x)
        x = Activation('relu')(x)
        return MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            data_format=self.channel_format
        )(x)

    def add_anchor_pred_layer(self, inputs):
        """
        添加锚框预测层
        """
        layer_anchor_count = inputs.shape[1] * inputs.shape[2] * self.anchor_count_per_unit
        self.all_anchor_count += layer_anchor_count
        cls_pred = Convolution2D(filters=self.anchor_count_per_unit * (self.class_count + 1),
                                 kernel_size=(3, 3),
                                 padding="same")(inputs)
        cls_pred = softmax(cls_pred)
        box_offset_pred = Convolution2D(filters=self.anchor_count_per_unit * 4,
                                        kernel_size=(3, 3),
                                        padding="same"
                                        )(inputs)
        return layer_anchor_count, cls_pred, box_offset_pred

    def combine_all_pred_layers(self):
        """
        合并所有的anchor_pred_layers层
        anchor_pred_layers： list of (cls_pred, box_offset_pred)
        cls_pred: shape [batch_size, row, col, anchor_count, class+1]
        box_offset_pred： shape [batch_size, row, col, anchor_count * 4]

        :return:
            shape: [batch_size,all layer( row * col * anchor_count), class+1],
                    [batch_size,all layer( row * col * anchor_count), 4],
        """
        all_cls_pred = []
        all_box_offset_pred = []
        layer_cnt = 0
        for layer_anchor_count, cls_pred, box_offset_pred in self.anchor_pred_layers:
            layer_cnt += 1
            cls_pred = tf.reshape(cls_pred, (-1, layer_anchor_count * (self.class_count + 1)),
                                  name=f"layer_{layer_cnt}_cls_pred")
            all_cls_pred.append(cls_pred)
            box_offset_pred = tf.reshape(box_offset_pred, (-1, layer_anchor_count * 4),
                                         name=f"layer_{layer_cnt}_box_offset_pred")
            all_box_offset_pred.append(box_offset_pred)
        all_cls_pred = tf.concat(all_cls_pred, axis=1, name="concat_all_layer_cls_pred")
        all_cls_pred = tf.reshape(all_cls_pred, (-1, self.all_anchor_count, self.class_count + 1), name="all_cls_pred")
        all_box_offset_pred = tf.concat(all_box_offset_pred, axis=1, name="concat_all_layer_offset_pred")
        all_box_offset_pred = tf.reshape(all_box_offset_pred, (-1, self.all_anchor_count, 4),
                                         name="all_box_offset_pred")
        return [all_cls_pred, all_box_offset_pred]

    def save_model(self, model_path: str = "./data/tiny_ssd.h5"):
        file_util.prepare_parent_folder(model_path)
        self.model.save(model_path)

    def load_model(self, model_path: str = "./data/tiny_ssd.h5"):
        self.model = keras.models.load_model(model_path)

    def predict(self, images):
        m = DetectAnchorBoxMatcher()
        batch_size = len(images)
        all_cls_pred, all_box_offset_pred = self.model.predict(np.array(images) / 256.0)
        res = []
        for i in range(batch_size):
            predict_boxes = []
            predict_probs = []
            predict_classes = []
            for j in range(len(self.all_anchors)):
                cls_pred = all_cls_pred[i][j]
                box_offset_red = all_box_offset_pred[i][j]
                box = m.anchor_to_label_box(to_box(self.all_anchors[j]), box_offset_red)
                cls = int(np.argmax(cls_pred))
                if cls < 0:
                    # 背景
                    continue
                if np.max(cls_pred) < 0.9:
                    continue
                predict_classes.append(cls)
                predict_boxes.append(box)
                predict_probs.append(np.max(cls_pred))
            res.append(
                m.nms_predict_boxes(predict_boxes, predict_probs, predict_classes)
            )
        return res


class SsdTrainer(object):

    def __init__(self, tiny_ssd: TinySSD = None):
        self.layer_params = LAYER_PARAMS
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.train_inputs = None
        self.train_outputs = None
        self.train_anchors = None
        self.test_inputs = None
        self.test_outputs = None
        self.test_anchors = None
        self.image_height = 256
        self.image_width = 256
        self.tiny_ssd = tiny_ssd

    def prepare_datasets(self):
        from datasets import D2LBananaDatasets
        print("loading D2LBananaDatasets...")
        d2l_banana_ds = D2LBananaDatasets()
        self.train_images, self.train_labels = d2l_banana_ds.load_train_dataset()
        self.test_images, self.test_labels = d2l_banana_ds.load_test_dataset()
        print("transforming D2LBananaDatasets...")
        self.train_inputs = np.array(self.train_images) / 256.0
        self.train_anchors, self.train_outputs = self.convert_to_outputs(self.train_labels)
        self.test_inputs = np.array(self.test_images) / 256.0
        self.test_anchors, self.test_outputs = self.convert_to_outputs(self.test_labels)
        print("prepare datasets ok.")
        print(f"train input shape: {self.train_inputs.shape} output shape"
              f" {self.train_outputs[0].shape}, {self.train_outputs[1].shape}")
        print(f"test input shape: {self.test_inputs.shape} output shape"
              f" {self.test_outputs[0].shape}, {self.test_outputs[1].shape}")

    def convert_to_outputs(self, labels):
        from anchor_box import gen_multi_layer_anchor_sample, to_box
        batch_size = len(labels)
        outputs = [
            [], [], []
        ]
        anchors_list = []
        for i in range(batch_size):
            label_boxes = [
                to_box(labels[i])
            ]
            anchors, box_offsets, box_masks, class_labels = gen_multi_layer_anchor_sample(self.image_height,
                                                                                          self.image_width, label_boxes,
                                                                                          self.layer_params)
            anchors_list.append(anchors)
            outputs[0].append(class_labels)
            outputs[1].append(box_offsets)
            outputs[2].append(box_masks)
        for i in range(len(outputs)):
            outputs[i] = np.array(outputs[i])
        return np.array(anchors_list), outputs[: -1]

    def train(self, epoch=128, batch_size=32):
        log_dir = "./log"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        if self.train_inputs is None or self.train_outputs is None:
            self.prepare_datasets()
        if self.tiny_ssd is None:
            self.tiny_ssd = TinySSD()
            self.tiny_ssd.build_net()
        self.tiny_ssd.model.fit(self.train_inputs, self.train_outputs, batch_size=batch_size, epochs=epoch,
                                callbacks=[tensorboard_callback])
