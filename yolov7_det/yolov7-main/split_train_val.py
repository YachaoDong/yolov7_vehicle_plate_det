
import os
import random
import argparse
import glob


if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir_path', type=str, default='/home/data/smoke_data/images/', help='input img label path')
    parser.add_argument('--save_dir_path', type=str, default='/home/data/smoke_data/', help='input img label path')
    
    opt = parser.parse_args()

    # 获取每个img的绝对路径
    abs_img_paths = glob.glob(opt.img_dir_path + "*.jpg")
    random.shuffle(abs_img_paths)
    num_imgs = len(abs_img_paths)
    train_percent = 0.8
    train_abs_img_paths = abs_img_paths[:int(train_percent*num_imgs)]
    test_abs_img_paths = abs_img_paths[int(train_percent*num_imgs):]
    
    with open(opt.save_dir_path + 'train.txt', 'w') as f1:
        for train_pwd in train_abs_img_paths:
            f1.write(train_pwd + '\n')
    
    with open(opt.save_dir_path + 'test.txt', 'w') as f1:
        for test_pwd in test_abs_img_paths:
            f1.write(test_pwd + '\n')

    with open(opt.save_dir_path + 'val.txt', 'w') as f1:
        for val_pwd in test_abs_img_paths:
            f1.write(val_pwd + '\n')


