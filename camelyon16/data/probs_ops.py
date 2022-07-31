"""
包含对概率热图的操作
主要是特征提取
"""
import cv2
import numpy as np
import scipy.stats.stats as st

from skimage.measure import label
from skimage.measure import regionprops
from openslide import OpenSlide
from openslide import OpenSlideUnsupportedFormatError

MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS = 0, 1, 2, 3, 4


class extractor_features(object):
    def __init__(self, probs_map, slide_path):
        """
        读入热图和原图
        """
        self._probs_map = probs_map
        self._slide = get_image_open(slide_path)

    def get_region_props(self, probs_map_threshold):
        """
        返回每个连通区域的属性
        """
        labeled_img = label(probs_map_threshold) # 标记连通区域
        return regionprops(labeled_img, intensity_image=self._probs_map) # 标注图像属性，返回一个列表，其每个项都包含图像的一个连通区域的属性

    def probs_map_set_p(self, threshold):
        """
        对热图进行基于阈值的二值化操作
        """
        probs_map_threshold = np.array(self._probs_map)

        probs_map_threshold[probs_map_threshold < threshold] = 0
        probs_map_threshold[probs_map_threshold >= threshold] = 1

        return probs_map_threshold

    def get_num_probs_region(self, region_probs):
        """
        连通区域的数量
        """
        return len(region_probs)

    def get_tumor_region_to_tissue_ratio(self, region_props):
        """
        返回概率热图中肿瘤区域与全部组织区域的比值
        """
        tissue_area = cv2.countNonZero(self._slide)
        tumor_area = 0

        n_regions = len(region_props)
        for index in range(n_regions):
            tumor_area += region_props[index]['area']

        return float(tumor_area) / tissue_area

    def get_largest_tumor_index(self, region_props):
        """
        返回热图中最大的肿瘤区域
        """
        largest_tumor_index = -1

        largest_tumor_area = -1

        n_regions = len(region_props)
        for index in range(n_regions):
            if region_props[index]['area'] > largest_tumor_area:
                largest_tumor_area = region_props[index]['area']
                largest_tumor_index = index

        return largest_tumor_index

    def f_area_largest_tumor_region_t50(self):
        pass

    def get_longest_axis_in_largest_tumor_region(self,
                                                 region_props,
                                                 largest_tumor_region_index):
        """
        返回最大肿瘤区域的最长轴
        """
        largest_tumor_region = region_props[largest_tumor_region_index]
        return max(largest_tumor_region['major_axis_length'],
                   largest_tumor_region['minor_axis_length'])

    def get_average_prediction_across_tumor_regions(self, region_props):
        """
        所有肿瘤区域的预测概率的平均值
        """
        # close 255
        region_mean_intensity = [region.mean_intensity for region in region_props]
        return np.mean(region_mean_intensity)

    def get_feature(self, region_props, n_region, feature_name):
        """
        返回图像特征
        返回一个长度为5的列表，分别是该属性的MAX MEAN VARIANCE SKEWNESS KURTOSIS
        """
        feature = [0] * 5
        if n_region > 0:
            feature_values = [region[feature_name] for region in region_props]
            feature[MAX] = format_2f(np.max(feature_values))
            feature[MEAN] = format_2f(np.mean(feature_values))
            feature[VARIANCE] = format_2f(np.var(feature_values))
            feature[SKEWNESS] = format_2f(st.skew(np.array(feature_values)))
            feature[KURTOSIS] = format_2f(st.kurtosis(np.array(feature_values)))

        return feature


def format_2f(number):
    return float("{0:.2f}".format(number))


def get_image_open(wsi_path):
    """
    打开 WSI 文件
    """
    try:
        wsi_image = OpenSlide(wsi_path)
        # 使用 WSI 图像降采样最高的level
        level_used = wsi_image.level_count - 1
        rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
        wsi_image.close()
    except OpenSlideUnsupportedFormatError:
        raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

    # 对图像进行二值化处理，将图像中颜色在范围区间内的像素设为1，其余为0
    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 对图像进行形态学变换
    # 闭运算：先膨胀，再腐蚀，主要用于清除细小黑点
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    # 开运算：先腐蚀，再膨胀，主要用于清除小亮点
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return image_open