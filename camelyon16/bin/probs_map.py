"""
生成概率热图
"""
import sys
import os
import argparse
import logging
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '\\..\\..\\')

from camelyon16.data.wsi_producer import WSIPatchDataset  # 见 ../data/wsi_producer.py

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# num_workers 不为 0 时会出现错误，尚不清楚原因

parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('mask_path', default=None, metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                    'workers to use to make batch, default 0')
parser.add_argument('--eight_avg', default=False, type=bool, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')


def chose_model(mod):
    if mod == 'resnet18':
        model = models.resnet18(weights=None)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(model, dataloader):
    """
    生成概率热图
    """
    probs_map = np.zeros(
        dataloader.dataset._mask.shape)  # 热图大小等同于掩膜大小（某个level）
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        # 计算掩膜上每个点对应的 patch 的概率
        with torch.no_grad():
            data = Variable(data.cuda(non_blocking=True))
        output = model(data)
        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of data is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        if len(output.shape) == 1:
            probs = output.sigmoid().cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           :].sigmoid().cpu().data.numpy().flatten()
        probs_map[x_mask, y_mask] = probs
        # # 用于打印 log 信息
        # count += 1
        # time_spent = time.time() - time_now
        # time_now = time.time()
        # logging.info(
        #     '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
        #     .format(
        #         time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
        #         dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, cnn, flip='NONE', rotate='NONE'):
    """
    生成 dataloader
    """
    batch_size = cnn['batch_size'] * 2
    num_workers = args.num_workers

    dataloader = DataLoader(
        WSIPatchDataset(args.wsi_path, args.mask_path,
                        image_size=cnn['image_size'],
                        crop_size=cnn['crop_size'], normalize=True,
                        flip=flip, rotate=rotate),  # 该类具体见 ../data/wsi_priducer.py
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)

    mask = np.load(args.mask_path)  # 载入组织掩膜
    ckpt = torch.load(args.ckpt_path)  # 载入 torch 模型数据

    model = chose_model(cnn['model'])
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 1)
    model.load_state_dict(ckpt['state_dict'])  # 载入训练的模型参数
    model = model.cuda().eval()

    if not args.eight_avg:
        # 参数 eight_avg 为假时原样计算概率热图
        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(model, dataloader)
    else:
        # 参数 eight_avg 为真时概率热图的每个patch的值等于8个旋转翻转方向预测值的平均
        probs_map = np.zeros(mask.shape)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        probs_map /= 8

    np.save(args.probs_map_path, probs_map)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
