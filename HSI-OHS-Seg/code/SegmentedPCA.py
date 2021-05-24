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

#  保存tif文件函数
def writeTiff(im_data, path, im_geotrans=(0,0,0,0,0,0), im_proj=""):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset
    
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
                PCA_label_dir, tif_num, width, height, suffix, Agu=False):
    # (n*w*h,4) -> (n,w,h,4)
    tif_datas_PCA = tif_datas_PCA.reshape((tif_num,
                                           width,
                                           height,
                                           4))
    
    if not os.path.exists(PCA_image_dir):
        os.makedirs(PCA_image_dir)
    if not os.path.exists(PCA_label_dir):
        os.makedirs(PCA_label_dir)
        
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
                PCA_tif_path = tif_paths[i].replace("images","seg_pca_images")
                PCA_tif_path = PCA_tif_path.replace(".tif",suffix+".tif")
                cv2.imwrite(PCA_tif_path, tif_data_PCA)
                label_path_source = label_paths[i]
                label_path_target = PCA_tif_path.replace("images","labels")
                copyfile(label_path_source, label_path_target)
            else:
                # 否则有1/2的概率进行增强
                ok = random.randint(1,3)
                if(ok>1):
                    tif_data_PCA = tif_datas_PCA[i]
                    PCA_tif_path = tif_paths[i].replace("images","seg_pca_images")
                    PCA_tif_path = PCA_tif_path.replace(".tif",suffix+".tif")
                    cv2.imwrite(PCA_tif_path, tif_data_PCA)
                    label_path_source = label_paths[i]
                    label_path_target = PCA_tif_path.replace("images","labels")
                    copyfile(label_path_source, label_path_target)
        else:
            tif_data_PCA = tif_datas_PCA[i]
            PCA_tif_path = tif_paths[i].replace("images","seg_pca_images")
            PCA_tif_path = PCA_tif_path.replace(".tif",suffix+".tif") 
            cv2.imwrite(PCA_tif_path, tif_data_PCA)
            # 如果是测试数据的话,没有标签
            if(len(label_paths)>0):
                label_path_source = label_paths[i]
                label_path_target = PCA_tif_path.replace("images","labels")
                copyfile(label_path_source, label_path_target)


# 每5个数据取第fold_k个数据为验证集
fold_k = 0

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

start = 0
ok = True
segmented_index = [0]
for i in range(channel):
    if(ok):
        ok = False
    else:
        corr = np.corrcoef(train_val_tif_datas[:,start],train_val_tif_datas[:,i])
        if(corr[0][1]<0.95):
            print(corr[0][1],start,i)
            segmented_index.append(i)
            ok = True
            start = i+1
segmented_index.append(channel)



train_val_tif_datas_segmenteds = []
train_tif_datas_segmenteds = []
val_tif_datas_segmenteds = []
test_tif_datas_segmenteds = []
for i in range(len(segmented_index)-1):
    print(segmented_index[i],segmented_index[i+1])
    train_val_tif_datas_segmented = train_val_tif_datas[:,segmented_index[i]:segmented_index[i+1]]
    train_val_tif_datas_segmenteds.append(train_val_tif_datas_segmented)
    train_tif_datas_segmented = train_tif_datas[:,segmented_index[i]:segmented_index[i+1]]
    train_tif_datas_segmenteds.append(train_tif_datas_segmented)
    val_tif_datas_segmented = val_tif_datas[:,segmented_index[i]:segmented_index[i+1]]
    val_tif_datas_segmenteds.append(val_tif_datas_segmented)
    test_tif_datas_segmented = test_tif_datas[:,segmented_index[i]:segmented_index[i+1]]
    test_tif_datas_segmenteds.append(test_tif_datas_segmented)


train_tif_datas_segmented_PCAs = []
val_tif_datas_segmented_PCAs = []
test_tif_datas_segmented_PCAs = []
for i in range(len(segmented_index)-1):
    # 取前PCA_n个成分
    PCA_n = 1
    # 生成PCA转换规则
    pca = PCA(n_components=PCA_n).fit(train_val_tif_datas_segmenteds[i])
    # 将规则应用于训练数据
    train_tif_datas_segmented_PCA = pca.transform(train_tif_datas_segmenteds[i])
    train_tif_datas_segmented_PCAs.append(train_tif_datas_segmented_PCA)
    # 将规则应用于验证数据
    val_tif_datas_segmented_PCA = pca.transform(val_tif_datas_segmenteds[i])
    val_tif_datas_segmented_PCAs.append(val_tif_datas_segmented_PCA)
    # 将规则应用于测试集
    test_tif_datas_segmented_PCA = pca.transform(test_tif_datas_segmenteds[i])
    test_tif_datas_segmented_PCAs.append(test_tif_datas_segmented_PCA)

train_tif_datas_segmented_PCAs = np.concatenate(train_tif_datas_segmented_PCAs, axis=1)
val_tif_datas_segmented_PCAs = np.concatenate(val_tif_datas_segmented_PCAs, axis=1)
test_tif_datas_segmented_PCAs = np.concatenate(test_tif_datas_segmented_PCAs, axis=1)



# 转成8bit
PCA_max = np.max([np.max(train_tif_datas_segmented_PCAs), 
                  np.max(val_tif_datas_segmented_PCAs)])
PCA_min = np.min([np.min(train_tif_datas_segmented_PCAs), 
                  np.min(val_tif_datas_segmented_PCAs)])

train_tif_datas_segmented_PCAs = ((train_tif_datas_segmented_PCAs - PCA_min)/
                                  (PCA_max - PCA_min) * 255).astype(np.uint8)
val_tif_datas_segmented_PCAs = ((val_tif_datas_segmented_PCAs - PCA_min)/
                                (PCA_max - PCA_min) * 255).astype(np.uint8)
test_tif_datas_segmented_PCAs = ((test_tif_datas_segmented_PCAs - PCA_min)/
                                (PCA_max - PCA_min) * 255).astype(np.uint8)

Seg_PCA_train_image_dir = "../data/train_{0}/seg_pca_images".format(fold_k)
Seg_PCA_train_label_dir = "../data/train_{0}/seg_pca_labels".format(fold_k)
Seg_PCA_val_image_dir = "../data/val_{0}/seg_pca_images".format(fold_k)
Seg_PCA_val_label_dir = "../data/val_{0}/seg_pca_labels".format(fold_k)
Seg_PCA_test_image_dir = "../data/test/seg_pca_images"
Seg_PCA_test_label_dir = "../data/test/seg_pca_labels"
 
SavePCAData(train_tif_datas_segmented_PCAs, train_tif_paths, train_label_paths, 
            Seg_PCA_train_image_dir, Seg_PCA_train_label_dir, train_tif_num, 
            width, height, suffix="")
SavePCAData(val_tif_datas_segmented_PCAs, val_tif_paths, val_label_paths,
            Seg_PCA_val_image_dir, Seg_PCA_val_label_dir, val_tif_num, 
            width, height, suffix="")
SavePCAData(test_tif_datas_segmented_PCAs, test_tif_paths, [],
            Seg_PCA_test_image_dir, Seg_PCA_test_label_dir, test_tif_num, 
            test_width, test_height, suffix="")


# # 训练数据增强
# for i in tqdm(range(channel)):
#     index = []
#     index_seg = []
#     for j in range(channel+1):
#         if(j in segmented_index):
#             if(j!=0):
#                 index.append(index_seg)
#                 index_seg = []
#             if(j!=i):
#                 index_seg = [j]
#         elif(j!=i):
#             index_seg.append(j)
#     # print(index)
#     train_tif_datas_segmenteds = []
#     for k in range(len(index)):
#         train_tif_datas_segmented = train_tif_datas[:,index[k]]
#         train_tif_datas_segmenteds.append(train_tif_datas_segmented)

#     train_tif_datas_segmented_PCAs = []
#     for k in range(len(index)):
#         # 取前PCA_n个成分
#         PCA_n = 1
#         # 生成PCA转换规则
#         pca = PCA(n_components=PCA_n).fit(train_tif_datas_segmenteds[k])
#         # 将规则应用于训练数据
#         train_tif_datas_segmented_PCA = pca.transform(train_tif_datas_segmenteds[k])
#         train_tif_datas_segmented_PCAs.append(train_tif_datas_segmented_PCA)
    
#     train_tif_datas_segmented_PCAs = np.concatenate(train_tif_datas_segmented_PCAs, axis=1)
  
#     # 转成8bit
#     PCA_max = np.max(train_tif_datas_segmented_PCAs)
#     PCA_min = np.min(train_tif_datas_segmented_PCAs)
    
#     train_tif_datas_segmented_PCAs = ((train_tif_datas_segmented_PCAs - PCA_min)/
#                                       (PCA_max - PCA_min) * 255).astype(np.uint8)
#     SavePCAData(train_tif_datas_segmented_PCAs, train_tif_paths, train_label_paths, 
#                 Seg_PCA_train_image_dir, Seg_PCA_train_label_dir, train_tif_num, 
#                 width, height, suffix="_"+str(i),Agu=True)