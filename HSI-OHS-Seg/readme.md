# [2021珠海数据开放创新应用大赛](http://www.zhzwfwdc.com/zodiac/index.html)-高光谱地物分类

## 1 赛题描述

    题目：“珠海一号”高光谱影像地物分类
    描述：针对“珠海一号”高光谱影像，对植被(1)、建筑(2)、裸地(3)、水体(4)四种地物进行分类。
    算法评价标准：Kappa系数

## 2 解决方案

我们采用PCA-UnetPlusPlus分割。

### 2.1 数据预处理

#### 2.1.1 统计不同类别的个数

我们先统计一下数据中的不同类别的分布情况，发现类别严重不均衡，需要添加diceloss以及针对性的数据增强来缓解这种情况。

![不同类别的个数](https://github.com/WangZhenqing-RS/2021ZODIAC/blob/main/HSI-OHS-Seg/plt/%E5%9C%B0%E7%89%A9%E8%A6%81%E7%B4%A0%E7%B1%BB%E5%88%AB%E5%83%8F%E7%B4%A0%E6%95%B0%E7%9B%AE%E5%9B%BE.png  "不同类别的个数")

#### 2.1.2 k-Fold分隔训练集和验证集

我们取k=5，将原始所有训练集分成5份，每取其中一份作为验证集，其余四份作为训练集。这样可以得到5折不同的训练集和验证集，最终用于模型集成，最大程度使用任务所给的数据。

#### 2.1.3 数据清洗

数据清洗，即去除对分类不利的数据。经过我们目视解译，发现32个波段中，第1个波段和第32个波段具有明显的噪声，故将其剔除。

#### 2.1.4 PCA降维

高光谱图像虽然光谱信息丰富，但是也存在着极大的信息冗余。并且在训练数据不是很充足的情况下，如果所有的波段全部参与训练很容易产生过拟合。所以我们采用PCA对数据进行降维处理。

##### 2.1.4.1 普通PCA降维

普通PCA降维是指对原始数据进行真正的PCA变换。前6个主成分的信息量有99.25%，各主成分的信息量如下表所示。我们分别实验了取前2-6个主成分进行以resnet18为backbone的UnetPlusPlus模型语义分割，验证集精度如下图所示，可知取4个主成分时精度最高。

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
| :--: | :--:  | :--: |
| 1 | 0.5701 | 0.5701 |
| 2 | 0.4051 | 0.9752 |
| 3 | 0.0115 | 0.9867 |
| 4 | 0.0031 | 0.9898 |
| 5 | 0.0018 | 0.9916 |
| 6 | 0.0009 | 0.9925 |


![不同主成分的精度](https://github.com/WangZhenqing-RS/2021ZODIAC/blob/main/HSI-OHS-Seg/plt/%E4%B8%8D%E5%90%8C%E4%B8%BB%E6%88%90%E5%88%86%E9%AA%8C%E8%AF%81%E9%9B%86IoU.png  "不同主成分的精度")

##### 2.1.4.2 分段PCA降维(segmented PCA)

普通PCA降维虽然实现了图像降维，但是也改变了原始数据的物理意义，使图像解译变得困难，而且当波段之间的相关性较弱的时候，也不宜进行PCA特征提取。基于此，我们同时进行了分段PCA的图像降维，我们将相关性很大的波段分成一组，然后分别对每组波段进行PCA。这样一来，避免了在全局主成分变换中因为变换的全局性使得局部重要的波段被忽略的缺点。实验中相关性阈值我们设置为0.95。

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

#### 2.1.5 数据增强

训练数据只有90 * 5 / 4 = 72张，这是远远不够的，所以要进行数据增强。我们采取线下剔除波段后主成分增强与线上旋转增强结合策略。首先将30个通道的数据中其中一个通道剔除后，对剩下的29个通道做主成分变换，为了缓解类别不均衡现象，我们对裸土占比>1%且背景占比<50%的图像(共5张)进行概率为1的上述操作。对其他的图像进行概率p=1/2的上述操作。这样一来可以得到约72 + 5 * 30 + 67 * 15 = 1092张图像。然后在线上进行随机水平镜像、垂直镜像、对角镜像数据增强。

之所以选择p=1/2,是因为如果p太小，模型会将拟合焦点过多的放在上述的那5张图像上，而如果p太大，类别不平衡则得不到缓解。下面的一个对比实验(训练集:验证集=4:1)也证明了1/2的合理性。

| p | val_mIoU | val_kappa |
| :--: | :--: | :--: |
| 1/3 | 0.779 | 0.806 |
| 1/2 | 0.784 | 0.812 |
| 2/3 | 0.773 | 0.800 |

### 2.2 模型构建

#### 2.2.1 UnetPlusPlus

我们采用UnetPlusPlus作为我们的任务总体模型框架，编码器采用ResnSt101，并在解码器中添加attention模块scSE，scSE是综合了通道维度和空间维度的注意力模块，可以增强有意义的特征，抑制无用特征，从而导致精度提升。

#### 2.2.2 损失函数

软交叉熵函数是对标签值进行标签平滑之后再与预测值做交叉熵计算，可以在一定程度上提高泛化性。diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定。我们采用软交叉熵函数和diceloss的联合函数作为实验的损失函数。

	# 损失函数采用SoftCrossEntropyLoss+DiceLoss
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    DiceLoss_fn=DiceLoss(mode='multiclass')
    # 软交叉熵,即使用了标签平滑的交叉熵,会增加泛化性
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    loss_fn = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                          first_weight=0.5, second_weight=0.5).cuda()

#### 2.2.3 优化器与学习率调整

我们选择Adamw优化器，初始学习率lr=1e-3，权重衰减weight_decay=1e-3。

	optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-3)

在训练时梯度下降算法可能陷入局部最小值，此时可以通过突然提高学习率，来“跳出”局部最小值并找到通向全局最小值的路径。所以我们采用余弦退火策略调整学习率。T_0=2，T_mult=2，eta_min=1e-5。

	# 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=2, # T_0就是初始restart的epoch数目
            T_mult=2, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
            eta_min=1e-5 # 最低学习率
            ) 


### 2.3 预测

#### 2.3.1 k折融合

将k折训练得到的k个模型预测结果进行平均，得到k折融合结果。

#### 2.3.2 不同模型融合

对PCA-UnetPlusPlus模型结果和Seg_PCA-UnetPlusPlus模型结果进行平均，得到不同模型融合结果。

#### 2.3.3 TTA测试增强

测试时对原图像、水平翻转图像、垂直翻转图像的预测结果进行平均，得到TTA结果。
