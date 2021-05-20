# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:22:41 2021

@author: DELL
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

#  初始化每个类的数目
vegetation_num = 0
building_num = 0
baresoil_num = 0
water_num = 0
others_num = 0

label_paths = glob.glob('../data/train/labels/*.tif')

for label_path in label_paths:
    label = cv2.imread(label_path,0)
    vegetation_num += np.sum(label == 1)
    building_num += np.sum(label == 2)
    baresoil_num += np.sum(label == 3)
    water_num += np.sum(label == 4)
    others_num += np.sum(label == 255)

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

classes = (
    '植被', 
    '建筑',
    '裸土',
    '水体',
    '其他'
    )

numbers = [
    vegetation_num,
    building_num,
    baresoil_num,
    water_num,
    others_num
    ]

plt.barh(classes, numbers)
for i, v in enumerate(numbers):
    plt.text(v, i, str(round(v/label.size/len(label_paths)*100,1))+"%",verticalalignment ="center")

plt.title('地物要素类别像素数目')
# #设置坐标轴范围
plt.xlim((0, 1.2e7))
# plt.ylim((-2, 2))
#设置坐标轴名称
plt.xlabel('像素数目')
plt.ylabel('类别')

plt.savefig("../plt/地物要素类别像素数目图.png", dpi = 300, bbox_inches="tight")
plt.show()