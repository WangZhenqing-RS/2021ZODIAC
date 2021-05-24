# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:41:11 2021

@author: DELL
"""

import glob
from shutil import copyfile
import os

image_paths = glob.glob("../data/train/images/*.tif")
label_paths = glob.glob("../data/train/labels/*.tif")

# k fold交叉验证
k = 5
for fold in range(k):
    fold_train_image_dir = "../data/train_"+str(fold)+"/images"
    fold_train_label_dir = "../data/train_"+str(fold)+"/labels"
    fold_val_image_dir = "../data/val_"+str(fold)+"/images"
    fold_val_label_dir = "../data/val_"+str(fold)+"/labels"
    if not os.path.exists(fold_train_image_dir):
        os.makedirs(fold_train_image_dir)
    if not os.path.exists(fold_train_label_dir):
        os.makedirs(fold_train_label_dir)
    if not os.path.exists(fold_val_image_dir):
        os.makedirs(fold_val_image_dir)
    if not os.path.exists(fold_val_label_dir):
        os.makedirs(fold_val_label_dir)
    for i in range(len(image_paths)):
        # 训练验证4:1,即每5个数据的第val_index个数据为验证集
        if i % 5 == fold:
            image_path = image_paths[i]
            fold_val_image_path = fold_val_image_dir + image_path.split("images")[-1]
            copyfile(image_path, fold_val_image_path)
            label_path = label_paths[i]
            fold_val_label_path = fold_val_label_dir + label_path.split("labels")[-1]
            copyfile(label_path, fold_val_label_path)
        else:
            image_path = image_paths[i]
            fold_train_image_path = fold_train_image_dir + image_path.split("images")[-1]
            copyfile(image_path, fold_train_image_path)
            label_path = label_paths[i]
            fold_train_label_path = fold_train_label_dir + label_path.split("labels")[-1]
            copyfile(label_path, fold_train_label_path)

