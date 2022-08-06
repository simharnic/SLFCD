"""
获取组织的掩码
也就是说，去除掉背景的空白部分，使用了基于阈值的图像分割，采用 OSTU 算法计算自适应阈值
"""
import sys
import os
import argparse
import logging

# 对于 Windows 平台，需要下载 openslide 的二进制版本并将 bin 文件夹路径添加到环境变量 'openslide' 中
import platform
if platform.system() == 'Windows':
    with os.add_dll_directory(os.getenv('openslide')):
        import openslide
else:
    import openslide

import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '\\..\\..\\')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')


def run(args):
    logging.basicConfig(level=logging.INFO)

    slide = openslide.OpenSlide(args.wsi_path)

    # img_RGB 的矩阵形状是 slide.level_dimensions 的转置
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                           args.level,
                           slide.level_dimensions[args.level]).convert('RGB')),
                           axes=[1, 0, 2])

    # RGB 转 HSV 因为 HSV 更适合进行颜色处理
    img_HSV = rgb2hsv(img_RGB)

    # 使用 OSTU 算法进行基于阈值的图像分割
    # RGB 下的图像遮罩
    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)

    # HSV 下的图像遮罩
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])

    # 由参数 RGB_min 手动给定的阈值
    min_R = img_RGB[:, :, 0] > args.RGB_min
    min_G = img_RGB[:, :, 1] > args.RGB_min
    min_B = img_RGB[:, :, 2] > args.RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

    np.save(args.npy_path, tissue_mask)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
