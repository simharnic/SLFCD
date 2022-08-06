"""
用于对训练集和验证集的图片进行处理生成dataset
包含了数据增强部分包括：色彩增强，随机旋转和翻转
虽然有 crop_size 参数但似乎并没有用到
"""
import os
import sys

import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)

from torchvision import transforms  # noqa


class ImageDataset(Dataset):

    def __init__(self, data_path, img_size,
                 crop_size=224, normalize=True):
        self._data_path = data_path
        self._img_size = img_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._pre_process()

    def _pre_process(self):
        """
        获取给定路径下的图片
        """
        # 获取给定目录下的目录作为分类 classes
        if sys.version_info >= (3, 5):
            # Faster and available in python 3.5 and above
            classes = [d.name for d in os.scandir(self._data_path) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._data_path) if os.path.isdir(os.path.join(self._data_path, d))]
        # 将目录分类 classes 转化为 index
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        # 根据目录分类 classes 确定数据标签，获取目录下所有 png 图片来生成 dataset
        self._items = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(self._data_path, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.split('.')[-1] == 'png':
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        self._items.append(item)

        random.shuffle(self._items)

        self._num_images = len(self._items)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path, label = self._items[idx]
        label = np.array(label, dtype=float)

        img = Image.open(path)

        # 色彩增强
        img = self._color_jitter(img)

        # 随机左右翻转
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机旋转
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)

        # PIL image to torch image
        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        # 归一化 [-1,1]
        if self._normalize:
            img = (img - 128.0) / 128.0

        return img, label

