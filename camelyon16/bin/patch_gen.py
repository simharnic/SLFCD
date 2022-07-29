"""
基于给定的坐标列表生成图像块数据集

当生成的中心点为 Tumor_Normal(肿瘤WSI中的正常区域) 时可能会导致生成的块上包含肿瘤区域
对结果的影响未知（原作者说不大）
"""
import sys
import os
import argparse
import logging
import time
from shutil import copyfile
# multiprocessing 用于进行多进程处理
from multiprocessing import Pool, Value, Lock

import openslide

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input directory of WSI files')
parser.add_argument('coords_path', default=None, metavar='COORDS_PATH',
                    type=str, help='Path to the input list of coordinates')
parser.add_argument('patch_path', default=None, metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
                    'generate patches, default 0')
parser.add_argument('--num_process', default=5, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opts):
    i, pid, x_center, y_center, args = opts
    # 计算块起始位置
    x = int(int(x_center) - args.patch_size / 2)
    y = int(int(y_center) - args.patch_size / 2)
    # 读取切片文件
    wsi_path = os.path.join(args.wsi_path, pid + '.tif')
    slide = openslide.OpenSlide(wsi_path)
    # 读取计算出的块的区域
    img = slide.read_region(
        (x, y), args.level,
        (args.patch_size, args.patch_size)).convert('RGB')
    # 保存块
    img.save(os.path.join(args.patch_path, str(i) + '.png'))
    
    global lock
    global count
    # lock 用于多进程间的同步
    # 一旦一个线程获得一个锁，会阻塞随后尝试获得锁的线程，直到它被释放；任何线程都可以释放它。
    # count 用于计数生成的块的数量，每生成100个块打印一次 log 信息。
    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(args):
    logging.basicConfig(level=logging.INFO)
    # 如果输出路径不存在则创建目录
    if not os.path.exists(args.patch_path):
        os.mkdir(args.patch_path)

    copyfile(args.coords_path, os.path.join(args.patch_path, 'list.txt'))
    # 生成需要处理的块的列表
    opts_list = []
    infile = open(args.coords_path)
    for i, line in enumerate(infile):
        pid, x_center, y_center = line.strip('\n').split(',')
        opts_list.append((i, pid, x_center, y_center, args))
    infile.close()
    # Pool 多进程处理
    pool = Pool(processes=args.num_process)
    pool.map(process, opts_list)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()