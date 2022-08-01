主要对 /camelyon16/ 文件夹下的代码进行了注释，这些代码使用 torch 实现 patch 分类；
/extras/下的代码我还没有仔细查看，应当是对应的 tensorflow 和 keras 的实现版本

目前尚未对代码进行运行测试，仅仅修改了代码中一些明显的错误

每个文件的用途在原来的 README.md 中都有说明，这里按使用顺序概括一下。详细内容参看每个文件的注释。

注释按照如下的顺序进行的，在前面注释过的在后面如果没有必要不再添加注释。


`/camelyon16/bin/camelyon16xml2json.py`：用来将 Camelyon16 组织官方提供的 xml 格式的肿瘤注释文件转换为 json 格式

`/camelyon16/data/annotation.py`：主要用于对肿瘤区域注释数据的相关处理

`/camelyon16/bin/tissue_mask.py`：获取组织的掩码

`/camelyon16/bin/tumor_mask.py`：获取癌症掩码

`/camelyon16/bin/non_tumor_mask.py`：获取癌症WSI中除去背景空白后正常区域的掩码

`/camelyon16/bin/sampled_spot_gen.py`：生成训练patch块的中心点

`/camelyon16/bin/patch_gen.py`：生成patch数据集

`/camelyon16/bin/train.py`：训练 CNN 模型

`/camelyon16/data/image_producer.py`：用于对训练集和验证集的图片进行处理生成dataset

`/camelyon16/bin/probs_map.py`：利用训练好的 CNN 模型生成概率热图

`/camelyon16/data/wsi_producer.py`：用于生成概率热图所需要的patch的dataset

`/camelyon16/bin/extract_feature_robsmap.py`：提取概率热图中的特征

`/camelyon16/data/probs_ops.py`：包含对概率热图的操作；主要是特征提取

`/camelyon16/bin/wsi_classification.py`：使用特征训练模型得到 WSI 分类器 以及对分类器进行 ROC 评估

`/camelyon16/bin/nms.py`：使用非最大抑制算法（nms）获得每个检测到的肿瘤区域在level 0 级的坐标

`/camelyon16/bin/Evaluation_FROC.py`：评估肿瘤定位的平均FROC评分
