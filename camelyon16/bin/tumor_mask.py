"""
获取癌症掩码
就是说，根据 Camelyon16 官方给的肿瘤注释文件得到肿瘤位置的掩码
"""
import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the JSON file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')


def run(args):

    # 得到参数指定的层级的 WSI 图像
    slide = openslide.OpenSlide(args.wsi_path)
    w, h = slide.level_dimensions[args.level]

    # 初始化掩膜
    mask_tumor = np.zeros((h, w))

    # 取得对应层级的降采样系数，如 level 6 就是 2^6
    factor = slide.level_downsamples[args.level]

    # 读取转化为 json 的肿瘤注释文件
    with open(args.json_path) as f:
        dicts = json.load(f)
    # 获取肿瘤多边形集合
    tumor_polygons = dicts['positive']

    for tumor_polygon in tumor_polygons:
        name = tumor_polygon["name"]
        # 获取多边形的顶点，除以对应的降采样系数得到实际位置
        vertices = np.array(tumor_polygon["vertices"]) / factor
        vertices = vertices.astype(np.int32)
        # 根据顶点将多边形在掩膜图像中填充绘制出来
        cv2.fillPoly(mask_tumor, [vertices], (255))
    # 将掩膜图像转化为二值
    mask_tumor = mask_tumor[:] > 127
    mask_tumor = np.transpose(mask_tumor) # 转置

    np.save(args.npy_path, mask_tumor)

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
