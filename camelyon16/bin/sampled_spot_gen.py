"""
生成块的中心点
也就是随机从给定的掩膜中提取点
"""
import os
import sys
import logging
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__), "\\..\\..\\"))

# 输出的 pid 跟 npy 一致而与 wsi 不一致导致生成 patch 时繁琐

parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
parser.add_argument("mask_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the mask npy file")
parser.add_argument("txt_path", default=None, metavar="TXT_PATH", type=str,
                    help="Path to the txt file")
parser.add_argument("patch_number", default=None, metavar="PATCH_NUMB", type=int,
                    help="The number of patches extracted from WSI")
parser.add_argument("--level", default=6, metavar="LEVEL", type=int,
                    help="Bool format, whether or not")


class patch_point_in_mask_gen(object):
    '''
    从掩膜中提取中心点
    输入: 掩膜路径, 需要的中心点数量
    输出: 中心点
    '''

    def __init__(self, mask_path, number):
        self.mask_path = mask_path
        self.number = number

    def get_patch_point(self):
        mask_tissue = np.load(self.mask_path)

        # 提取掩膜区域所有点的坐标
        X_idcs, Y_idcs = np.where(mask_tissue)
        centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)
        # 随机选取所需要的数量的中心点
        if centre_points.shape[0] > self.number:
            sampled_points = centre_points[np.random.randint(centre_points.shape[0],
                                                             size=self.number), :]
        else:
            sampled_points = centre_points
        return sampled_points


def run(args):
    # 选取中心点
    sampled_points = patch_point_in_mask_gen(
        args.mask_path, args.patch_number).get_patch_point()
    # 考虑对应层级的降采样系数，计算得到其在原图中的坐标
    sampled_points = (sampled_points * 2 ** args.level).astype(np.int32)
    # 导出
    mask_name = os.path.split(args.mask_path)[-1].split(".")[0]
    name = np.full((sampled_points.shape[0], 1), mask_name)
    center_points = np.hstack((name, sampled_points))
    txt_path = args.txt_path
    with open(txt_path, "a") as f:
        np.savetxt(f, center_points, fmt="%s", delimiter=",")


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
