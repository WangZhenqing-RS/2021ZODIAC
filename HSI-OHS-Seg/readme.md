# [2021珠海数据开放创新应用大赛](http://www.zhzwfwdc.com/zodiac/index.html)-高光谱地物分类

## 1.赛题描述

    题目：“珠海一号”高光谱影像地物分类
    描述：针对“珠海一号”高光谱影像，对裸地、植被、水体、建筑四种地物进行分类。
    算法评价标准：Kappa系数

## 2.解决方案

我们采用PCA降维+UnetPlusPlus分割。

### 2.1数据预处理

#### 2.1.1统计不同类别的个数

我们先统计一下数据中的不同类别的分布情况(如下图所示)，发现类别严重不均衡，需要添加diceloss以及上采样少类别来缓解这种情况。

![不同类别的个数](https://github.com/WangZhenqing-RS/2021ZODIAC/blob/main/HSI-OHS-Seg/plt/%E5%9C%B0%E7%89%A9%E8%A6%81%E7%B4%A0%E7%B1%BB%E5%88%AB%E5%83%8F%E7%B4%A0%E6%95%B0%E7%9B%AE%E5%9B%BE.png)


