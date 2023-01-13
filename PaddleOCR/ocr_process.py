# -*- coding: utf-8 -*-
'''
@Time    : 6/28/2022 4:29 PM
@Author  : dong.yachao
'''
# -*- coding: utf-8 -*-
import numpy as np

'''
@Time    : 6/7/2022 5:06 PM
@Author  : dong.yachao
'''

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import argparse
import glob

from pathlib import Path
import glob
import random
import numpy as np
import pandas as pd
import cv2 as cv
import time
from multiprocessing.pool import Pool
from tqdm import tqdm
from itertools import repeat

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

# box: [0,6], line:[6], key_point:[7, 14]
classes = ['truck', 'van', 'car', 'slagcar', 'bus', 'fire_truck',
           'police_car', 'ambulance', 'SUV', 'microbus', 'unknown_vehicle',
           'plate', 'double_plate']

vehicle_color = ['white', 'silver', 'grey', 'black', 'red', 'blue', 'yellow', 'green', 'brown', 'others']

abs_path = os.getcwd()


def convert_annotation(xml_path='/project/train/src_repo/dataset/xmls/*.xml'):
    img_path = xml_path.replace('.xml', '.jpg')

    in_file = open(xml_path, encoding='utf-8')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # plate poly obj
    ocr_data = []
    for obj in root.iter('polygon'):
        # difficult = obj.find('difficult').text

        # 获取车牌内容
        plate_text = None
        for attribute in obj.find('attributes').iter('attribute'):
            if attribute.find('name').text == 'ocr':
                plate_text = attribute.find('value').text

        # 获取bbox
        points = obj.find('points').text
        points = points.split(';')  # ["x1,y1", "x2,y3", "x3,y3","x4,y4"]
        poly = []  # [xyxyxyxy]
        for i in points:
            x = float(i.split(',')[0])
            y = float(i.split(',')[1])
            poly.append(x)
            poly.append(y)

        poly_x, poly_y = np.array(poly).reshape(-1, 2).T
        poly = [poly_x.min(), poly_y.min(), poly_x.max(), poly_y.max()]

        ocr_data.append((img_path, str(plate_text), poly))

    return ocr_data


def split_txt(txt_path, train_percent=0.8):
    # 生成训练测试集，写入txt文件中
    with open(txt_path, 'r', encoding='utf-8') as f1:
        all_lines = f1.readlines()
        random.shuffle(all_lines)
        num_lines = len(all_lines)

        train_lines = all_lines[:int(train_percent * num_lines)]
        test_lines = all_lines[int(train_percent * num_lines):]

        with open(join(os.path.dirname(txt_path), 'ocr_train.txt'), 'w', encoding='utf-8') as f2:
            for train_pwd in train_lines:
                f2.write(train_pwd)

        with open(join(os.path.dirname(txt_path), 'ocr_test.txt'), 'w', encoding='utf-8') as f3:
            for test_pwd in test_lines:
                f3.write(test_pwd)

        with open(join(os.path.dirname(txt_path), 'ocr_val.txt'), 'w', encoding='utf-8') as f4:
            for val_pwd in test_lines:
                f4.write(val_pwd)


def compute_pixs(bbox):
    # xyxy
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return (w, h, w * h)


def crop_and_count(args):
    # cur_ocr_data: (img_abs_path, plate_content, bbox)
    cur_ocr_data, save_crop_img_dir, i = args
    img_abs_path, plate_content, plate_box = cur_ocr_data

    ori_img = cv.imread(img_abs_path)

    w, h, wh = compute_pixs(plate_box)

    # 统计为None的像素
    if plate_content == 'NONE':

        return [[w, h, wh], None, None]
    else:
        wavy_count = plate_content.count('~')
        underline_count = plate_content.count('_')
        # save crop img xyxy
        crop_img = ori_img[int(plate_box[1]):int(plate_box[3]), int(plate_box[0]):int(plate_box[2])]
        crop_img_path = join(save_crop_img_dir, os.path.basename(img_abs_path).split('.')[0] + f'_{i}.jpg')
        cv.imwrite(crop_img_path, crop_img)
        # none_count, all_count, f.write
        return [None, [w, h, wh, wavy_count, underline_count], crop_img_path + '\t' + plate_content + '\n']


def crop_img_and_count_plate(all_ocr_data, ocr_data_path, use_vague=False):
    '''

    @param all_ocr_data: # [(img_abs_path, plate_content, box), ...]
    @return:
    '''

    save_crop_img_dir = join(ocr_data_path, 'ocr_crop_imgs')
    os.makedirs(save_crop_img_dir, exist_ok=True)

    ocr_txt_dir = ocr_data_path

    none_count = []
    all_count = []  # 除去None的个数统计

    with open(join(ocr_txt_dir, 'all_ocr.txt'), 'w', encoding='utf-8') as f:
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(crop_and_count, zip(all_ocr_data, repeat(save_crop_img_dir), range(len(all_ocr_data)))),
                        total=len(all_ocr_data))
            for none_content, all_content, f_content in pbar:
                # 统计为None的像素
                if none_content is not None:
                    none_count.append(none_content)
                else:
                    all_count.append(all_content)
                    if use_vague:
                        f.write(f_content)
                    else:
                        if (all_content[-1] == 0) and (all_content[-2] == 0):
                            f.write(f_content)

    # 分割all_ocr.txt数据集
    split_txt(join(ocr_txt_dir, 'all_ocr.txt'), train_percent=0.8)

    # 统计分析
    # [[w, h, wh]...]
    if none_count:
        none_df = pd.DataFrame(data=np.array(none_count), columns=['w', 'h', 'wh'])
        none_info = none_df.describe()

        none_df.to_csv(join(ocr_txt_dir, 'none_df.csv'))

        print('none_info:\n', none_info)

    # [[w, h, wh, ~count, _count]...]
    if all_count:
        all_df = pd.DataFrame(data=np.array(all_count),
                              columns=['w', 'h', 'wh', 'wavy_count', 'underline_count'])

        all_df.to_csv(join(ocr_txt_dir, 'all_df.csv'))

        all_info = all_df.describe()

        print('all_info:\n', all_info)

        print('all_info wavy_count w:\n', all_df.groupby(['wavy_count'])['w'].describe())
        print('all_info wavy_count h:\n', all_df.groupby(['wavy_count'])['h'].describe())
        print('all_info wavy_count wh:\n', all_df.groupby(['wavy_count'])['wh'].describe())

        print('all_info underline_count w:\n', all_df.groupby(['underline_count'])['w'].describe())
        print('all_info underline_count h:\n', all_df.groupby(['underline_count'])['h'].describe())
        print('all_info underline_count wh:\n', all_df.groupby(['underline_count'])['wh'].describe())

    print('none_count:', len(none_count))
    print('all_count:', len(all_count))


if __name__ == '__main__':
    start_time = time.time()

    # 获取所有的img路径
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_txt', type=str, default=r'/home/data/ocr_data/all_det_imgs.txt',
                        help='img所在文件目录路径')

    parser.add_argument('--ocr_data_dir', type=str,
                        default=r'/home/data/ocr_data',
                        help='需要保存ocr_data文件目录路径')

    parser.add_argument('--use_vague', type=bool,
                        default=False,
                        help='')

    opt = parser.parse_args()

    with open(opt.img_path_txt, 'r') as f:
        img_abs_path = [i.strip() for i in f.readlines()]

    # ocr_data/:
    #                ocr_crop_imgs/
    #                all_ocr.txt
    #                train_ocr.txt
    #                test_ocr.txt
    # 读取plate points，转换为bbox  xxx.txt: plate, x y x y

    all_ocr_data = []
    for xml in img_abs_path:
        xml = xml.replace('.jpg', '.xml')
        # image_name = xml.strip().split(os.sep)[-1].split('.')[0]
        # [(img_path, plate_content, box), ...]
        cur_ocr_data = convert_annotation(xml_path=xml.strip())
        for i in cur_ocr_data:
            all_ocr_data.append(i)

    # 统计数据
    # /data,
    opt.use_vague = False
    crop_img_and_count_plate(all_ocr_data, opt.ocr_data_dir, use_vague=opt.use_vague)
    print('cost time:', time.time() - start_time)
