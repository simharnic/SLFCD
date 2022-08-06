"""
用来将 Camelyon16 组织官方提供的 xml 格式的肿瘤注释文件转换为 json 格式
"""
import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '\\..\\..\\')

from camelyon16.data.annotation import Formatter  # 见 ../data/annotation.py

# 使用 argparse 包
# argparse.ArgumentParser() 和 add_argument() 用于指定输入参数以及提供描述
# run(args) 可以看作实际运行的“主函数”
# 之后所有 py 文件都是如此，不再进行注释
parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')
parser.add_argument('xml_path', default=None, metavar='XML_PATH', type=str,
                    help='Path to the input Camelyon16 xml annotation')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the output annotation in json format')


def run(args):
    Formatter.camelyon16xml2json(args.xml_path, args.json_path)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
