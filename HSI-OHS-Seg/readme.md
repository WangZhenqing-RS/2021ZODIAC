# [2021珠海数据开放创新应用大赛](http://www.zhzwfwdc.com/zodiac/index.html)-高光谱地物分类

## 1.赛题描述

    题目：“珠海一号”高光谱影像地物分类
    描述：针对“珠海一号”高光谱影像，对裸地、植被、水体、建筑四种地物进行分类。
    算法评价标准：Kappa系数

## 2.解决方案

我们采用PCA降维+UnetPlusPlus分割。

### 2.1数据预处理

#### 2.1.1统计不同类别的个数

我们先统计一下数据中的不同类别的分布情况(如图1所示)，发现类别严重不均衡，需要添加diceloss以及上采样少类别来缓解这种情况。

![不同类别的个数](https://github.com/WangZhenqing-RS/2021ZODIAC/blob/main/HSI-OHS-Seg/plt/%E5%9C%B0%E7%89%A9%E8%A6%81%E7%B4%A0%E7%B1%BB%E5%88%AB%E5%83%8F%E7%B4%A0%E6%95%B0%E7%9B%AE%E5%9B%BE.png  "不同类别的个数")

#### 2.1.2PCA降维

高光谱图像虽然光谱信息丰富，但是也存在着极大的信息冗余。并且在训练数据不是很充足的情况下，如果所有的波段全部参与训练很容易产生过拟合。所以我们采用PCA对数据进行降维处理。前6个主成分的信息量有99.25%，各主成分的信息量如下表所示。我们分别实验了取前2-6个主成分进行语义分割，验证集精度如下图所示，取4个主成分时精度最高。

	from sklearn.decomposition import PCA
    # 取前PCA_n个成分
    PCA_n = 6
    # 生成PCA转换规则
    pca = PCA(n_components=PCA_n).fit(train_tif_datas)
    # 将规则应用于训练数据
    train_tif_datas_PCA = pca.transform(train_tif_datas)
    # 将规则应用于测试集
    test_tif_datas_PCA = pca.transform(test_tif_datas)

| 主成分 | 信息量 | 累积信息量 |
| :-----| :----- | :----- |
| 1 | 0.5701 | 0.5701 |
| 2 | 0.4051 | 0.9752 |
| 3 | 0.0115 | 0.9867 |
| 4 | 0.0031 | 0.9898 |
| 5 | 0.0018 | 0.9916 |
| 6 | 0.0009 | 0.9925 |


![不同主成分的精度](https://github.com/WangZhenqing-RS/2021ZODIAC/blob/main/HSI-OHS-Seg/plt/%E4%B8%8D%E5%90%8C%E4%B8%BB%E6%88%90%E5%88%86%E9%AA%8C%E8%AF%81%E9%9B%86IoU.png  "不同主成分的精度")
#### 2.1.3
