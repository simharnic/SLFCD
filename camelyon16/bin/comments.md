每个文件的用途在原来的 README.md 中都有说明，这里按使用顺序概括一下。详细内容参看每个文件的注释。

注释按照如下的顺序进行的，在前面注释过的在后面如果没有必要不再添加注释。

`camelyon16xml2json.py`：用来将 Camelyon16 组织官方提供的 xml 格式的肿瘤注释文件转换为 json 格式

`tissue_mask.py`：获取组织的掩码

`tumor_mask.py`：获取癌症掩码

`non_tumor_mask.py`：获取癌症WSI中除去背景正常区域的掩码

`sampled_spot_gen.py`：获取随机的癌症区域中的坐标点

`patch_gen.py`：生成图像块数据集

`train.py`：训练 CNN 模型

`probs_map.py`：利用训练好的 CNN 模型生成概率热图

`extract_feature_robsmap.py`：提取概率热图中的特征

`wsi_classification.py`：使用特征训练模型得到 WSI 分类器 以及对分类器进行 ROC 评估

`nms.py`：使用非最大抑制算法（nms）获得每个检测到的肿瘤区域在level 0 级的坐标

`Evaluation_FROC.py`：评估肿瘤定位的平均FROC评分
