# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:02:21 2021

@author: DELL
"""

import numpy as np
try:
    import gdal
except:
    from osgeo import gdal
import os
import glob
import cv2
from sklearn.metrics import cohen_kappa_score

""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""  
def imgread(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    return data

def ConfusionMatrix(numClass, imgPredict, Label):  
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return precision  

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):  
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def kappa(confusionMatrix):
    # 返回kappa系数
    pe_rows = np.sum(confusionMatrix, axis=0)
    pe_cols = np.sum(confusionMatrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusionMatrix) / float(sum_total)
    return (po - pe) / (1 - pe)


#################################################################
#  标签图像文件夹
label_folder = r"E:\WangZhenQing\2021ZODIAC\hsi\data\test_old\labels\*.tif"
#  预测图像文件夹
pred_floder = r"E:\WangZhenQing\2021ZODIAC\hsi\data\test_old\pred\*.tif"
#  类别数目(包括背景)
classNum = 5
#  类别颜色字典
colorDict_GRAY = [255,1,2,3,4]

label_paths = glob.glob(label_folder)
pred_paths = glob.glob(pred_floder)

#  读取第一个图像，后面要用到它的shape
Label0 = imgread(label_paths[0])

#  图像数目
label_num = len(label_paths)

#  把所有图像放在一个数组里
label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
predict_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
for i in range(label_num):
    Label = imgread(label_paths[i])
    label_all[i] = Label
    Predict = imgread(pred_paths[i])
    predict_all[i] = Predict

#  拉直成一维
label_all = label_all.flatten()
predict_all = predict_all.flatten()

label_all[label_all==255] = 0
predict_all[predict_all==255] = 0 

#  计算混淆矩阵及各精度参数
confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
precision = Precision(confusionMatrix)
recall = Recall(confusionMatrix)
OA = OverallAccuracy(confusionMatrix)
IoU = IntersectionOverUnion(confusionMatrix)
FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
mIOU = MeanIntersectionOverUnion(confusionMatrix)
f1ccore = F1Score(confusionMatrix)
kappa = kappa(confusionMatrix)

for i in range(len(colorDict_GRAY)):
    print(colorDict_GRAY[i], end = "  ")
print("")
print("混淆矩阵:")
print(confusionMatrix)
print("精确度:")
print(precision)
print("召回率:")
print(recall)
print("F1-Score:")
print(f1ccore)
print("整体精度:")
print(OA)
print("IoU:")
print(IoU)
print("mIoU:")
print(mIOU)
print("FWIoU:")
print(FWIOU)
print("kappa:")
print(kappa)