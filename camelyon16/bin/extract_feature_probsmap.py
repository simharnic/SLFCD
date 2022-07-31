"""
提取概率热图中的特征
"""
import os
import sys
import logging
import argparse
import openslide
import numpy as np
import pandas as pd
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../../'))

from camelyon16.data.probs_ops import extractor_features # 见 ../data/probs_ops.py

parser = argparse.ArgumentParser(description='Extract features from probability map'
                                             'for slide classification')
parser.add_argument('probs_map_path', default=None, metavar='PROBS MAP PATH',
                    type=str, help='Path to the npy probability map ')
parser.add_argument('wsi_path', default=None, metavar='WSI PATH',
                    type=str, help='Path to the whole slide image. ')
parser.add_argument('feature_path', default=None, metavar='FEATURE PATH',
                    type=str, help='Path to the output features file')

probs_map_feature_names = ['region_count', 'ratio_tumor_tissue', 'largest_tumor_area', 'longest_axis_largest_tumor',
                           'pixels_gt_90', 'avg_prediction', 'max_area', 'mean_area', 'area_variance', 'area_skew',
                           'area_kurt', 'max_perimeter', 'mean_perimeter', 'perimeter_variance', 'perimeter_skew',
                           'perimeter_kurt', 'max_eccentricity', 'mean_eccentricity', 'eccentricity_variance',
                           'eccentricity_skew', 'eccentricity_kurt', 'max_extent', 'mean_extent', 'extent_variance',
                           'extent_skew', 'extent_kurt', 'max_solidity', 'mean_solidity', 'solidity_variance',
                           'solidity_skew', 'solidity_kurt', 'label'] # 并未使用？


def compute_features(extractor):
    features = []  # 总的特征，为对每个图像都长度固定的列表
    # 分别设置概率热图的阈值为 0.9 和 0.5
    # 高于阈值的部分被视为肿瘤区域
    probs_map_threshold_p90 = extractor.probs_map_set_p(0.9)
    probs_map_threshold_p50 = extractor.probs_map_set_p(0.5)

    # 获取不同阈值下每个肿瘤区域的区域信息
    region_props_p90 = extractor.get_region_props(probs_map_threshold_p90)
    region_props_p50 = extractor.get_region_props(probs_map_threshold_p50)

    # p=0.9 肿瘤区域数量
    f_count_tumor_region = extractor.get_num_probs_region(region_props_p90)  # feature列表第1项
    features.append(f_count_tumor_region)
    # p=0.9 肿瘤区域占全部组织的比值
    f_percentage_tumor_over_tissue_region = extractor.get_tumor_region_to_tissue_ratio(region_props_p90)  # 第2项
    features.append(f_percentage_tumor_over_tissue_region)
    # p=0.5 最大肿瘤区域的索引
    largest_tumor_region_index_t50 = extractor.get_largest_tumor_index(region_props_p50)
    # p=0.5 最大肿瘤区域的面积
    f_area_largest_tumor_region_t50 = region_props_p50[largest_tumor_region_index_t50].area  # 3
    features.append(f_area_largest_tumor_region_t50)
    # p=0.5 最大肿瘤区域的最长轴
    f_longest_axis_largest_tumor_region_t50 = extractor.get_longest_axis_in_largest_tumor_region(region_props_p50,
                                                                                                 largest_tumor_region_index_t50)  # 4
    features.append(f_longest_axis_largest_tumor_region_t50)
    # p=0.9 所有肿瘤区域的面积
    f_pixels_count_prob_gt_90 = cv2.countNonZero(probs_map_threshold_p90)  # 5
    features.append(f_pixels_count_prob_gt_90)
    # p=0.9 所有肿瘤区域的平均预测概率
    f_avg_prediction_across_tumor_regions = extractor.get_average_prediction_across_tumor_regions(region_props_p90)  # 6
    features.append(f_avg_prediction_across_tumor_regions)
    # p=0.9 肿瘤区域的 'area'面积属性 的 MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS
    f_area = extractor.get_feature(region_props_p90, f_count_tumor_region, 'area')  # 7,8,9,10,11
    features += f_area
    # p=0.9 肿瘤区域的 'perimeter'周长属性
    f_perimeter = extractor.get_feature(region_props_p90, f_count_tumor_region, 'perimeter')  # 12,13,14,15,16
    features += f_perimeter
    # p=0.9 'eccentricity'偏心率
    f_eccentricity = extractor.get_feature(region_props_p90, f_count_tumor_region, 'eccentricity')  # 17,18,19,20,21
    features += f_eccentricity
    # p=0.5 'exten'等于area除以其最小边界矩形面积
    f_extent_t50 = extractor.get_feature(region_props_p50, len(region_props_p50), 'extent')  # 22,23,24,25,26
    features += f_extent_t50
    # p=0.9 'solidity'等于area除以其最小外接凸多边形convex面积
    f_solidity = extractor.get_feature(region_props_p90, f_count_tumor_region, 'solidity')  # 27,28,29,30,31 总共31个特征
    features += f_solidity

    return features


def run(args):
    # 读入图像
    slide_path = args.wsi_path
    probs_map = np.load(args.probs_map_path)
    # 特征提取器，见 ../data/image_producer.py
    extractor = extractor_features(probs_map, slide_path)
    # 计算特征
    features = compute_features(extractor)
    # 根据 WSI 文件名判断是否有肿瘤，作为最后一个即第32个特征
    wsi_name = os.path.split(args.wsi_path)[-1]
    if 'umor' in wsi_name:
        features += [1]
    elif 'ormal' in wsi_name:
        features += [0]
    else:
        features += ['Nan']
    # 转换为 DataFrame 并输出
    df = (pd.DataFrame(data=features)).T
    df.to_csv(args.feature_path, index=False, sep=',')


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
