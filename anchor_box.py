#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: anchor_box
@author: jkguo
@create: 2023/1/4
"""
import typing
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

eps = 1e-8


def to_category(arr, cls_count):
    res = []
    for item in arr:
        rc = [0.0] * cls_count
        rc[item] = 1.0
        res.append(rc)
    return np.array(res)


class Box(object):

    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def center(self):
        return (self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0

    def size(self):
        """
        :return: (width, height)
        """
        return (self.max_x - self.min_x), (self.max_y - self.min_y)

    def area(self) -> float:
        """
        计算锚框面积
        :return:
        """
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def calc_iou(self, box) -> float:
        """
        计算两个锚框或边界框列表中成对的交并比
        :param box:
        :return:
        """
        # 计算两个锚框的交集
        i_x1 = max(self.min_x, box.min_x)
        i_x2 = min(self.max_x, box.max_x)
        i_y1 = max(self.min_y, box.min_y)
        i_y2 = min(self.max_y, box.max_y)
        # 如果没有交集，则iou为0
        if i_x2 <= i_x1 or i_y2 <= i_y1:
            return 0.0
        # 并集面积
        i_area = (i_x2 - i_x1) * (i_y2 - i_y1)
        # 补集面积
        u_area = self.area() + box.area() - i_area
        return i_area / u_area

    def draw(self, color="w"):
        rect = patches.Rectangle((self.min_x, self.min_y),
                                 self.max_x - self.min_x,
                                 self.max_y - self.min_y,
                                 linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)

    def __str__(self):
        return f"({self.min_x}, {self.min_y}, {self.max_x}, {self.max_y})"


def to_box(rec: typing.Tuple) -> Box:
    return Box(rec[0], rec[1], rec[2], rec[3])


class DetectAnchorBoxGenerator(object):

    def __init__(self):
        pass

    @staticmethod
    def generate(height, width, feature_row_count: int, feature_col_count: int, scale: float,
                 aspect_ratio: float) -> typing.List:
        """
        自动生成锚框
        :param height:  图像的高度
        :param width:  图像的宽度
        :param feature_row_count: 目标结果的行数
        :param feature_col_count: 目标结果的列数
        :param scale: 锚框相对图像的高度的比例
        :param aspect_ratio: 锚框的高度和宽度的比例
        :return: 生成的锚框。锚框个数为： feature_row_count * feature_col_count
            shape: [feature_row_count][feature_col_count]
            item: (min_x, min_y, max_x, max_y)
        """
        boxes = []
        sqt_aspect_ratio = math.sqrt(aspect_ratio)
        stride_x = width * 1.0 / feature_col_count
        stride_y = height * 1.0 / feature_row_count
        for i in range(feature_row_count):
            b_r = []
            y = (i + 0.5) * stride_y
            for j in range(feature_col_count):
                x = (j + 0.5) * stride_x
                min_x = round(x - height * scale * sqt_aspect_ratio / 2)
                max_x = round(x + height * scale * sqt_aspect_ratio / 2)
                min_y = round(y - height * scale / sqt_aspect_ratio / 2)
                max_y = round(y + height * scale / sqt_aspect_ratio / 2)
                b_r.append((min_x, min_y, max_x, max_y))
            boxes.append(b_r)
        return boxes

    def batch_generate(self, height: int, width: int, feature_row_count: int, feature_col_count: int, scales: list,
                       aspect_ratios: list):
        """
        批量生成锚框
        :param height:  图像的高度
        :param width:  图像的宽度
        :param feature_row_count: 目标结果的行数
        :param feature_col_count: 目标结果的列数
        :param scales: 锚框相对图像的高度的比例列表
        :param aspect_ratios: 锚框的高度和宽度的比例列表
        :return: 生成的锚框。锚框个数为： feature_row_count * feature_col_count * (len(scales) + len(aspect_ratios) -1 )
            shape: [feature_row_count][feature_col_count][len(scales) + len(aspect_ratios) -1 ]
            item: (min_x, min_y, max_x, max_y)
        """
        boxes = []
        for scale in scales:
            boxes.append(self.generate(height, width, feature_row_count, feature_col_count, scale, aspect_ratios[0]))
        if len(aspect_ratios) > 1:
            for ratio in aspect_ratios[1:]:
                boxes.append(self.generate(height, width, feature_row_count, feature_col_count, scales[0], ratio))
        return np.transpose(np.array(boxes), (1, 2, 0, 3))


class DetectAnchorBoxMatcher(object):

    def __init__(self, match_iou_threshold=0.3, nms_threshold=0.5):
        """
        :param match_iou_threshold: 在标注锚框时， 锚框和标注框的iou最小阈值
        :param nms_threshold: 在做预测时，对预测后，所有的预测边框进行mns合并抑制时，需要抑制的iou阈值
        """
        self.match_iou_threshold = match_iou_threshold
        self.nms_threshold = nms_threshold

    def match_label_boxes(self, anchors: typing.List[Box], label_boxes: typing.List[Box]):
        """
        为anchors中所有的锚框标注对应的列表，并且计算box offset偏移量
        :param anchors: 自动生成的锚框列表
        :param label_boxes: 标注了的边框列表
        :return: box_offsets, box_masks, class_labels
            为每个锚框标注对应匹配的边框，计算对应的box offset
            box_offsets: 锚框的偏移量将根据和中心坐标的相对位置以及这两个框的相对大小进行标记。 鉴于数据集内不同的框的位置和大小不同，我们可以对那些相对位置和大小应用变换，使其获得分布更均匀且易于拟合的偏移量。
            box_masks: 如果为0则表示匹配的是背景，如果为1则表示匹配的为匹配了边框
            class_labels: 锚框匹配对应的边框分类（对应边框label_boxes的下标+1）0表示背景框
        """
        box_offsets = [None] * len(anchors)
        class_labels = [0] * len(anchors)
        box_masks = [None] * len(anchors)
        iou_map = np.zeros((len(anchors), len(label_boxes)))
        for i in range(len(anchors)):
            for j in range(len(label_boxes)):
                iou_map[i][j] = anchors[i].calc_iou(label_boxes[j])
        max_iou_idx = np.argmax(iou_map, axis=1)
        for i in range(len(anchors)):
            idx = max_iou_idx[i]
            if iou_map[i][idx] < self.match_iou_threshold:
                class_labels[i] = 0
                box_masks[i] = [0.0] * 4
                box_offsets[i] = (0, 0, 0, 0)
            else:
                class_labels[i] = idx + 1
                box_masks[i] = [1.0] * 4
                box_offsets[i] = self.calc_box_offset(anchors[i], label_boxes[idx])
        return box_offsets, box_masks, class_labels

    def calc_box_offset(self, anchor: Box, label_box: Box) -> typing.Tuple:
        """
        针对锚框 anchor 计算和匹配的 label_box 边框计算偏移量
        :param anchor: 锚框box
        :param label_box: 标注的边框box
        :return: (x, y, w, h) 锚框的偏移量将根据和中心坐标的相对位置以及这两个框的相对大小进行标记。 鉴于数据集内不同的框的位置和大小不同，我们可以对那些相对位置和大小应用变换，使其获得分布更均匀且易于拟合的偏移量。
        """
        xa, ya = anchor.center()
        wa, ha = anchor.size()
        xb, yb = label_box.center()
        wb, hb = label_box.size()
        return 10 * (xb - xa) / wa, 10 * (yb - ya) / ha, 5 * (math.log(wb / wa + eps)), 5 * (math.log(hb / ha + eps))

    def anchor_to_label_box(self, anchor: Box, box_offset) -> Box:
        """
        根据锚框和偏移量，反向计算出标注框
        :param anchor: 锚框box
        :param box_offset: (x, y, w, h) 锚框的偏移量将根据和中心坐标的相对位置以及这两个框的相对大小进行标记。 鉴于数据集内不同的框的位置和大小不同，我们可以对那些相对位置和大小应用变换，使其获得分布更均匀且易于拟合的偏移量。
        :return: 标注边框
        """
        xa, ya = anchor.center()
        wa, ha = anchor.size()
        xo, yo, wo, ho = box_offset
        xb = 0.1 * xo * wa + xa
        yb = 0.1 * yo * ha + ya
        wb = math.pow(math.e, 0.2 * wo) * wa
        hb = math.pow(math.e, 0.2 * ho) * ha
        return Box(xb - wb / 2.0, yb - hb / 2.0, xb + wb / 2.0, yb + hb / 2.0)

    def nms_predict_boxes(self, predict_boxes: typing.List[Box], predict_probs: typing.List[float],
                          predict_classes: typing.List[int]):
        """
        用mns算法抑制预测的边框
        :param predict_boxes: 预测的边框
        :param predict_probs:
        :param predict_classes
        :return:
        """
        pick_boxes = []
        idxes = np.flip(np.argsort(predict_probs))
        for idx in idxes:
            cls = predict_classes[idx]
            box = predict_boxes[idx]
            ignore = False
            for pb, _ in pick_boxes:
                if box.calc_iou(pb) >= self.nms_threshold:
                    ignore = True
                    break
            if not ignore:
                pick_boxes.append((box, cls))
        return pick_boxes


def gen_all_anchors(height, width, layer_params: typing.List[typing.Dict]):
    """
    根据layers参数，生成所有的anchors
    :param height:  图像的高度
    :param width:  图像的宽度
    :param layer_params: 每层的参数 list of
        {
            "h",
            "w",
            "scales": []
            "aspect_ratios": []
        }
    :return:
        anchors: shape [anchor_count, 4]
    """
    g = DetectAnchorBoxGenerator()
    anchors_list = []
    for param in layer_params:
        anchors = g.batch_generate(height, width, param["h"], param["w"], param["scales"], param["aspect_ratios"])
        anchors_list.append(anchors.reshape((-1, 4)))
    return np.concatenate(anchors_list, axis=0)


def gen_multi_layer_anchor_sample(height, width, label_boxes: typing.List[Box], layer_params: typing.List[typing.Dict]):
    """
    根据图像样本，生成多层的锚框样本
    :param height:  图像的高度
    :param width:  图像的宽度
    :param label_boxes: 边框列表
    :param layer_params: 每层的参数 list of
        {
            "h",
            "w",
            "scales": []
            "aspect_ratios": []
        }
    :return:
        anchors: shape [anchor_count, 4]
        box_offsets: shape [anchor_count, 4]
        box_masks: shape [anchor_count, 4]
        class_labels: shape [anchor_count, cls_count + 1]
    """
    g = DetectAnchorBoxGenerator()
    m = DetectAnchorBoxMatcher()
    anchors_list = []
    box_offsets_list = []
    box_masks_list = []
    class_labels_list = []
    for param in layer_params:
        anchors = g.batch_generate(height, width, param["h"], param["w"], param["scales"], param["aspect_ratios"])
        anchor_boxes = [to_box(i) for i in anchors.reshape((-1, 4))]
        box_offsets, box_masks, class_labels = m.match_label_boxes(anchor_boxes, label_boxes)
        anchors_list.append(anchors.reshape((-1, 4)))
        box_offsets_list.append(np.array(box_offsets))
        box_masks_list.append(np.array(box_masks))
        class_labels = to_category(class_labels, len(label_boxes) + 1)
        class_labels_list.append(class_labels)
    return (
        np.concatenate(anchors_list, axis=0),
        np.concatenate(box_offsets_list, axis=0),
        np.concatenate(box_masks_list, axis=0),
        np.concatenate(class_labels_list, axis=0)
    )
