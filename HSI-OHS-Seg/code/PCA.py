# -*- coding: utf-8 -*-
"""
Created on Sun May  2 18:18:58 2021

@author: DELL
"""

from sklearn.decomposition import PCA
from osgeo import gdal
import glob
import numpy as np
import os
import cv2
from shutil import copyfile
from tqdm import tqdm
import random

#  读取图像像素矩阵
#  tif_path 图像路径
def ReadTif(tif_path):
    dataset = gdal.Open(tif_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    return data

def GetAllData(tif_paths):
    tif_num = len(tif_paths)
    tif_datas = []
    for tif_path in tif_paths:
        # 读取数据
        tif_data = ReadTif(tif_path)
        # (c,w,h) -> (w,h,c)
        tif_data = tif_data.swapaxes(1, 0).swapaxes(1, 2)
        tif_datas.append(tif_data)
    
    width = tif_data.shape[0]
    height = tif_data.shape[1]
    channel = tif_data.shape[2]
        
    tif_datas = np.array(tif_datas)
    # (n,w,h,c) -> (n*w*h,c)
    tif_datas = tif_datas.reshape(
        (tif_num * width * height, channel)
        )
    
    # 去掉第一个波段和最后一个波段，这两个波段噪声明显
    tif_datas = tif_datas[:,1:channel-1]
    return tif_datas, tif_num, width, height, channel-2
    
def SavePCAData(tif_datas_PCA, tif_paths, label_paths, PCA_image_dir, 
                PCA_label_dir, tif_num, width, height, PCA_n, suffix, Agu=False):
    
    if not os.path.exists(PCA_image_dir):
        os.makedirs(PCA_image_dir)
    if not os.path.exists(PCA_label_dir):
        os.makedirs(PCA_label_dir)
    # (n*w*h,PCA_n) -> (n,w,h,PCA_n)
    tif_datas_PCA = tif_datas_PCA.reshape((tif_num,
                                           width,
                                           height,
                                           PCA_n))
    for i in range(tif_num):
        if(Agu): 
            label_path_source = label_paths[i]
            label = cv2.imread(label_path_source,0)
            land_sum = np.sum(label == 3)
            land_per = land_sum / label.size
            other_sum = np.sum(label == 255)
            other_per = other_sum / label.size
            # 如果裸地像素较多并且背景值较少
            if(land_per > 0.01 and other_per<0.5):
                tif_data_PCA = tif_datas_PCA[i]
                PCA_tif_path = tif_paths[i].replace("images","pca_images")
                PCA_tif_path = PCA_tif_path.replace(".tif",suffix+".tif")
                cv2.imwrite(PCA_tif_path, tif_data_PCA)
                label_path_target = PCA_tif_path.replace("images","labels")
                copyfile(label_path_source, label_path_target)
            else:
                # 否则有1/2的概率进行增强
                ok = random.randint(1,2)
                if(ok==1):
                    tif_data_PCA = tif_datas_PCA[i]
                    PCA_tif_path = tif_paths[i].replace("images","pca_images")
                    PCA_tif_path = PCA_tif_path.replace(".tif",suffix+".tif")
                    cv2.imwrite(PCA_tif_path, tif_data_PCA)
                    label_path_target = PCA_tif_path.replace("images","labels")
                    copyfile(label_path_source, label_path_target)
                
        else:
            tif_data_PCA = tif_datas_PCA[i]
            PCA_tif_path = tif_paths[i].replace("images","pca_images")
            PCA_tif_path = PCA_tif_path.replace(".tif",suffix+".tif")
            cv2.imwrite(PCA_tif_path, tif_data_PCA)
            # 如果是测试数据的话,没有标签
            if(len(label_paths)>0):
                label_path_source = label_paths[i]
                label_path_target = PCA_tif_path.replace("images","labels")
                copyfile(label_path_source, label_path_target)

# 每5个数据取第fold_k个数据为验证集
fold_k = 4
# 原始训练数据
train_val_tif_paths = glob.glob("../data/train/images/*.tif")
train_val_tif_datas, train_val_tif_num, width, height, channel = GetAllData(train_val_tif_paths)

# 训练数据
train_tif_paths = glob.glob("../data/train_{0}/images/*.tif".format(fold_k))
train_label_paths = glob.glob("../data/train_{0}/labels/*.tif".format(fold_k))
train_tif_datas, train_tif_num, width, height, channel = GetAllData(train_tif_paths)

# 验证数据
val_tif_paths = glob.glob("../data/val_{0}/images/*.tif".format(fold_k))
val_label_paths = glob.glob("../data/val_{0}/labels/*.tif".format(fold_k))
val_tif_datas, val_tif_num, width, height, channel = GetAllData(val_tif_paths)

# 测试数据
test_tif_paths = glob.glob("../data/test/images/*.tif")
test_tif_datas, test_tif_num, test_width, test_height, channel = GetAllData(test_tif_paths)

# 取前PCA_n个成分
PCA_n = 4
# 生成PCA转换规则
pca = PCA(n_components=PCA_n).fit(train_val_tif_datas)
# 将规则应用于训练数据
train_tif_datas_PCA = pca.transform(train_tif_datas)
# 将规则应用于验证数据
val_tif_datas_PCA = pca.transform(val_tif_datas)
# 将规则应用于测试数据
test_tif_datas_PCA = pca.transform(test_tif_datas)

# 转成8bit,方便后续处理
PCA_max = np.max([np.max(train_tif_datas_PCA),np.max(val_tif_datas_PCA)])
PCA_min = np.min([np.min(train_tif_datas_PCA),np.min(val_tif_datas_PCA)])
train_tif_datas_PCA = ((train_tif_datas_PCA - PCA_min)/
                       (PCA_max - PCA_min) * 255).astype(np.uint8)
val_tif_datas_PCA = ((val_tif_datas_PCA - PCA_min)/
                     (PCA_max - PCA_min) * 255).astype(np.uint8)
test_tif_datas_PCA = ((test_tif_datas_PCA - PCA_min)/
                      (PCA_max - PCA_min) * 255).astype(np.uint8)

PCA_train_image_dir = "../data/train_{0}/pca_images".format(fold_k)
PCA_train_label_dir = "../data/train_{0}/pca_labels".format(fold_k)
PCA_val_image_dir = "../data/val_{0}/pca_images".format(fold_k)
PCA_val_label_dir = "../data/val_{0}/pca_labels".format(fold_k)
PCA_test_image_dir = "../data/test/pca_images"
PCA_test_label_dir = "../data/test/pca_labels"
 
SavePCAData(train_tif_datas_PCA, train_tif_paths, train_label_paths, 
            PCA_train_image_dir, PCA_train_label_dir, train_tif_num, 
            width, height, PCA_n, suffix="")
SavePCAData(val_tif_datas_PCA, val_tif_paths, val_label_paths,
            PCA_val_image_dir, PCA_val_label_dir, val_tif_num, 
            width, height, PCA_n, suffix="")
SavePCAData(test_tif_datas_PCA, test_tif_paths, [],
            PCA_test_image_dir, PCA_test_label_dir, test_tif_num, 
            test_width, test_height, PCA_n, suffix="")

# 训练数据增强
for i in tqdm(range(channel)):
    index = []
    for j in range(channel):
        if(j!=i):
            index.append(j)
    # 删除第i个波段
    train_tif_datas_agu = train_tif_datas[:,index]
    # 生成PCA转换规则
    pca = PCA(n_components=PCA_n).fit(train_tif_datas_agu)
    # 将规则应用于训练数据
    train_tif_datas_PCA = pca.transform(train_tif_datas_agu)
    
    # 转成8bit,方便后续处理
    PCA_max = np.max(train_tif_datas_PCA)
    PCA_min = np.min(train_tif_datas_PCA)
    train_tif_datas_PCA = ((train_tif_datas_PCA - PCA_min)/
                           (PCA_max - PCA_min) * 255).astype(np.uint8)
    SavePCAData(train_tif_datas_PCA, train_tif_paths, train_label_paths, 
                PCA_train_image_dir, PCA_train_label_dir, train_tif_num, 
                width, height, PCA_n, suffix="_"+str(i), Agu=True)