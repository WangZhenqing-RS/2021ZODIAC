# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:04:28 2021

@author: DELL
"""

import torch.utils.data as D
from torchvision import transforms as T
import random
import numpy as np
import torch
import cv2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 



#  随机数据增强
#  image 图像
#  label 标签
def DataAugmentation(image, label, mode):
    if(image.shape[2]<=4):
        image = cv2.resize(image,(512,512))
    else:
        image1 = image[:,:,0:4]
        image1 = cv2.resize(image1,(512,512))
        image2 = image[:,:,4:image.shape[2]]
        image2 = cv2.resize(image2,(512,512))
        print(image1.shape)
        print(image2.shape)
        image = np.concatenate((image1,image2),axis=2)
    label = cv2.resize(label,(512,512),interpolation=cv2.INTER_NEAREST)
    if(mode == "train"):
        hor = random.choice([True, False])
        if(hor):
            #  图像水平翻转
            image = np.flip(image, axis = 1)
            label = np.flip(label, axis = 1)
        ver = random.choice([True, False])
        if(ver):
            #  图像垂直翻转
            image = np.flip(image, axis = 0)
            label = np.flip(label, axis = 0)
    return image, label

#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集Iou
def cal_val_iou(model, loader):
    val_iou = []
    # 需要加上model.eval()
    # 否则的话，有输入数据，即使不训练，它也会改变权值
    # 这是model中含有BN和Dropout所带来的的性质
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output = output.argmax(1)
        iou = cal_iou(output, target)
        val_iou.append(iou)
    return val_iou

# 计算IoU
def cal_iou(pred, mask, c=5):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p*t).sum()
        #  0.0001防止除零
        iou = 2*overlap/(uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)

class OurDataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_paths)
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index],cv2.IMREAD_UNCHANGED)
        if self.mode == "train":
            label = cv2.imread(self.label_paths[index],0)
            label[label == 255] = 0
            image, label = DataAugmentation(image, label, self.mode)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "val":
            label = cv2.imread(self.label_paths[index],0)
            label[label == 255] = 0
            # 常规来讲,验证集不需要数据增强,但是这次数据测试集和训练集不同域,为了模拟不同域,验证集也进行数据增强
            image, label = DataAugmentation(image, label, self.mode)
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "test":
            image = cv2.resize(image,(512,512))
            return self.as_tensor(image), self.image_paths[index]
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(image_paths, label_paths, mode, batch_size, 
                   shuffle, num_workers):
    dataset = OurDataset(image_paths, label_paths, mode)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

def split_train_val(image_paths, label_paths, val_index=0):
    # 分隔训练集和验证集
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = [], [], [], []
    for i in range(len(image_paths)):
        # 训练验证4:1,即每5个数据的第val_index个数据为验证集
        if i % 5 == val_index:
            val_image_paths.append(image_paths[i])
            val_label_paths.append(label_paths[i])
        else:
            train_image_paths.append(image_paths[i])
            train_label_paths.append(label_paths[i])
    print("Number of train images: ", len(train_image_paths))
    print("Number of val images: ", len(val_image_paths))
    return train_image_paths, train_label_paths, val_image_paths, val_label_paths
