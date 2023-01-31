# 物体检测学习工程

## 数据集下载

- 加载数据

```
from datasets import D2LBananaDatasets

d2l_banana_ds = D2LBananaDatasets("./data")

train_images, train_labels = d2l_banana_ds.load_train_dataset()
test_images, test_labels = d2l_banana_ds.load_test_dataset()
```

- 数据shape

```
train_images[0] shape
(256, 256, 3)
train_labels
[[104  20 143  58]
 [ 68 175 118 223]
 [163 173 218 239]
 ...
 [ 47  54  86 109]
 [ 43 125  90 166]
 [191  99 249 152]]
 ```

## 数据转换

- [目标检测锚框](目标检测锚框.ipynb)

## SSD目标检测

- [SSD-目标检测](SSD-目标检测.ipynb)

## 语义分割

- [语义分割数据集](VOC语义分割数据集.ipynb)
- [语义分隔识别物体](VOC语义分割.ipynb)
