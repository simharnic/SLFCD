"""
用于生成概率热图所需要的patch的dataset
包含了数据增强部分包括：随机旋转和翻转
虽然有 crop_size 参数但似乎并没有用到
"""
import os
import numpy as np
from torch.utils.data import Dataset
import PIL

# 对于 Windows 平台，需要下载 openslide 的二进制版本并将 bin 文件夹路径添加到环境变量 'openslide' 中
import platform
if platform.system() == 'Windows':
    with os.add_dll_directory(os.getenv('openslide')):
        import openslide
else:
    import openslide


class WSIPatchDataset(Dataset):

    def __init__(self, wsi_path, mask_path, image_size=256, crop_size=224,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._image_size = image_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        # 分别加载掩膜和切片文件
        self._mask = np.load(self._mask_path)
        self._slide = openslide.OpenSlide(self._wsi_path)
        # 加载掩膜和切片的大小，注意这里掩膜有可能对应 level 0 以外的其他 level
        X_slide, Y_slide = self._slide.level_dimensions[0]
        X_mask, Y_mask = self._mask.shape
        # 检验掩膜和切片的宽高比例是否一致
        if X_slide / X_mask != Y_slide / Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            'X_slide / X_mask: {} / {}, '
                            'Y_slide / Y_mask: {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))
        # 计算降采样系数，并检验其是否为 2 的幂
        self._resolution = X_slide * 1.0 / X_mask
        if not np.log2(self._resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2: '
                            '{}'.format(self._resolution))

        # 组织掩膜中所有的组织区域的坐标
        self._X_idcs, self._Y_idcs = np.where(self._mask)
        self._idcs_num = len(self._X_idcs)

    def __len__(self):
        return self._idcs_num # 掩膜中所有点的个数

    def __getitem__(self, idx):
        """
        取得由 idx 指定的掩膜中的点所对应的区域
        """
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        # x y 对应的原图中的中心点
        x_center = int((x_mask + 0.5) * self._resolution)
        y_center = int((y_mask + 0.5) * self._resolution)
        # 对应原图中的区域
        x = int(x_center - self._image_size / 2)
        y = int(y_center - self._image_size / 2)
        img = self._slide.read_region(
            (x, y), 0, (self._image_size, self._image_size)).convert('RGB')

        # 执行翻转和旋转操作
        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if self._rotate == 'ROTATE_90':
            img = img.transpose(PIL.Image.ROTATE_90)
        if self._rotate == 'ROTATE_180':
            img = img.transpose(PIL.Image.ROTATE_180)
        if self._rotate == 'ROTATE_270':
            img = img.transpose(PIL.Image.ROTATE_270)

        # 调整图像从 PIL 格式到 torch 格式
            # PIL image:   H x W x C
            # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        # 归一化 [-1,1]
        if self._normalize:
            img = (img - 128.0) / 128.0

        return (img, x_mask, y_mask)
