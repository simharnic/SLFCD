"""
训练模型
"""
import sys
import os
import argparse
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from torchvision import models
from torch import nn

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from camelyon16.data.image_producer import ImageDataset

# 设置随机数种子，便于复现结果
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=2, type=int, help='number of'
                    ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0', type=str, help='comma'
                    ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')


def chose_model(cnn):
    """
    选择模型，原项目只使用了 resnet18
    """
    if cnn['model'] == 'resnet18':
        model = models.resnet18(pretrained=False) # 加载未预训练(pretrained=False)的 resnet18 模型
    else:
        raise Exception("I have not add any models. ")
    return model


def train_epoch(summary, summary_writer, cnn, model, loss_fn, optimizer,
                dataloader_train):
    """
    训练模型，运行一个 epoch
    """
    model.train() # 将模型设为训练模式
    # 模型的训练和预测模式与 BatchNormalization 和 Dropout 层有关
    # Batch Normalization
    # 其作用对网络中间的每层进行归一化处理，并且使用变换重构（Batch Normalization Transform）保证每层提取的特征分布不会被破坏。
    # Dropout
    # 其作用为克服过拟合，在每个训练批次中，通过以一定概率忽略特征检测器，可以明显的减少过拟合现象。
    #
    # 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); batchnorm层会继续计算数据的mean和var等参数并更新。
    # 在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
    # 


    steps = len(dataloader_train) # 训练的步数，即batch数
    batch_size = dataloader_train.batch_size
    dataiter_train = iter(dataloader_train) # 创建迭代器

    time_now = time.time()
    for step in range(steps):
        data_train, target_train = next(dataiter_train) # 训练数据的输入和目标输出

        # 将训练数据装载到容器 Variable 并放置到显存中去
        data_train = Variable(data_train.float().cuda(async=True))
        target_train = Variable(target_train.float().cuda(async=True))

        # 取得当前模型的输出并计算损失函数
        output = model(data_train)
        output = torch.squeeze(output) # 维度压缩，删除所有大小为 1 的维
        loss = loss_fn(output, target_train)

        optimizer.zero_grad() # 清空过往梯度
        loss.backward() # 反向传播计算当前梯度
        optimizer.step() # 根据梯度更新网络参数

        probs = output.sigmoid() # 对输出应用 sigmoid 函数归一化为概率形式
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor) # 预测结果，大于等于 0.5 则为 1 反之为 0
        
        # 计算并输出 log 信息
        acc_data = (predicts == target_train).type(
            torch.cuda.FloatTensor).sum().data * 1.0 / batch_size # 准确率
        loss_data = loss.data # 损失率
        time_spent = time.time() - time_now # 时间花费
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                summary['step'] + 1, loss_data, acc_data, time_spent))
        summary['step'] += 1
        if summary['step'] % cnn['log_every'] == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])

    summary['epoch'] += 1

    return summary


def valid_epoch(summary, model, loss_fn,
                dataloader_valid):
    """
    验证模型，运行一个epoch
    """
    model.eval() # 模型设置为预测模式
    # 该模式不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行反传（backprobagation）
    # 而with torch.no_grad()则主要是用于停止autograd模块的工作，以起到加速和节省显存的作用

    steps = len(dataloader_valid)
    batch_size = dataloader_valid.batch_size
    dataiter_valid = iter(dataloader_valid)

    loss_sum = 0
    acc_sum = 0
    for step in range(steps):
        # 验证数据的输入输出
        data_valid, target_valid = next(dataiter_valid)
        data_valid = Variable(data_valid.float().cuda(async=True), volatile=True)
        target_valid = Variable(target_valid.float().cuda(async=True))

        # 计算损失函数
        output = model(data_valid)
        output = torch.squeeze(output) # important
        loss = loss_fn(output, target_valid)

        # 计算预测概率和值
        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
        # 计算准确率和损失率
        acc_data = (predicts == target_valid).type(
            torch.cuda.FloatTensor).sum().data * 1.0 / batch_size
        loss_data = loss.data

        loss_sum += loss_data
        acc_sum += acc_data
    # 平均准确率和损失率
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary


def run(args):
    # 载入 cnn 设置文件， 见 configs 文件夹下的 cnn.json
    with open(args.cnn_path, 'r') as f:
        cnn = json.load(f)
    # 如果输出目录不存在则创建
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # 将 cnn 设置写入输出目录下的 cnn.json 文件
    with open(os.path.join(args.save_path, 'cnn.json'), 'w') as f:
        json.dump(cnn, f, indent=1)

    # 指定训练时使用的 GPU，并由此确定训练中的 batch_size 和 workers 等参数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids # 指定程序可见的 GPU
    num_GPU = len(args.device_ids.split(','))
    batch_size_train = cnn['batch_size'] * num_GPU
    batch_size_valid = cnn['batch_size'] * num_GPU
    num_workers = args.num_workers * num_GPU
    # 模型初始化
    model = chose_model(cnn) # 选择模型
    fc_features = model.fc.in_features # 获取全连接层的输入个数
    model.fc = nn.Linear(fc_features, 1) # 设置全连接层（1是输出的个数）
    model = DataParallel(model, device_ids=None) # 指定模型层面的 GPU 并行处理，None 表示使用所有可见 GPU
    model = model.cuda() # 将模型加载到 GPU
    loss_fn = BCEWithLogitsLoss().cuda() # 损失函数
    optimizer = SGD(model.parameters(), lr=cnn['lr'], momentum=cnn['momentum']) # 优化器：随机梯度下降

    # 加载数据
    # dataset_train = ImageFolder(cnn['data_path_train'])
    # dataset_valid = ImageFolder(cnn['data_path_valid'])
    dataset_train = ImageDataset(cnn['data_path_train'],
                                 cnn['image_size'],
                                 cnn['crop_size'],
                                 cnn['normalize'])
    dataset_valid = ImageDataset(cnn['data_path_valid'],
                                 cnn['image_size'],
                                 cnn['crop_size'],
                                 cnn['normalize'])
    
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size_train,
                                  num_workers=num_workers)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=batch_size_valid,
                                  num_workers=num_workers)

    # 初始化
    # summary
    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(args.save_path)
    # 验证时的最小损失率，初始化为无穷大
    loss_valid_best = float('inf')
    for epoch in range(cnn['epoch']):
        # 进行一个 epoch 的训练和验证
        summary_train = train_epoch(summary_train, summary_writer, cnn, model,
                                    loss_fn, optimizer,
                                    dataloader_train) # 训练模型

        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt')) # 保存模型（最终为最后的模型）

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, model, loss_fn,
                                    dataloader_valid) # 验证模型
        time_spent = time.time() - time_now
        # 输出 log / summary 信息
        logging.info('{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, '
                     'Validation ACC: {:.3f}, Run Time: {:.2f}'
                     .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                             summary_train['step'], summary_valid['loss'],
                             summary_valid['acc'], time_spent))
        summary_writer.add_scalar('valid/loss',
                                  summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar('valid/acc',
                                  summary_valid['acc'], summary_train['step'])
        # 记录验证时的最小损失率，并保存
        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.module.state_dict()},
                       os.path.join(args.save_path, 'best.ckpt')) # 保存模型（最终为验证时具有最小损失率的模型）

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
