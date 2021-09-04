---
layout: post
title:  "CornerNet: Detecting Objects as Paired Keypoints(代码未添加)"
date:   2021-08-24 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/CornerNet>

CornerNet论文：

<https://arxiv.org/abs/1808.01244>

CornerNet-Lite论文：

<https://arxiv.org/abs/1904.08900>

官方CornerNet-Lite的pytorch代码：

<https://github.com/princeton-vl/CornerNet-Lite>

说明：本文为cornernet的理解。但代码看的是CornerNet-Lite，不太匹配。。。


## P1. 简介
该文提出CornerNet，一个不需要anchor的一阶段目标检测算法。将目标建模为目标框的左上角和右下角的角点，并使用CNN分别预测当前类别所有目标左上角和右下角角点的热图，以及每个角点的特征向量（embedding vector）。特征向量用于对相同目标的角点分组，训练网络来预测相同的特征向量，使得相同目标的两个角点的特征向量间的距离更小。该算法简化了网络的输出，不再需要设计anchor，整体框图如图1所示。

![1](/assets/post/2021-08-24-CornerNet/1cornernet.png)
_图1 cornernet_

该文还提出了corner pooling，一个新的池化层，帮助CNN更好的定位边界框的角点。边界框的角点通常是在目标外面（才能把目标框起来）。此时无法根据局部信息定位角点。因而为了确定某个位置是否是左上角的角点，需要水平向右看物体的最高边缘，同时垂直向下看物体的最左侧边缘。这促使我们提出了corner pooling层：其包含两个特征图，在每个位置其在第一个特征图取上取当前点到最右侧所有特征的最大值（max-pool），同时在第二个特征图上取当前点到最下侧所有特征的最大值（max-pool），最终把这两个池化结果相加。如图2所示。

![2](/assets/post/2021-08-24-CornerNet/2cornernetpooling.png)
_图2 cornernetpooling_

为何使用角点比使用边界框中心或候选框要好，作者给出了两个猜测的原因：① 由于目标框需要依赖目标的4个边界，因而目标框的中心比较难于定位；而定位角点只需要2个边界，因而更容易定位，corner pooling更是如此，其编码了一些角点的先验知识。② 角点更高效的提供了密集离散化边界框的方式：只需要
$$O\left( wh \right)$$
 个角点就能代表
$$O\left( { {w}^{2}}{ {h}^{2}} \right)$$
个可能的anchor。

## P2. CornerNet
### P2.1 简介
该算法将目标建模为目标框的左上角和右下角的角点，并使用CNN分别预测当前类别所有目标左上角和右下角角点的热图，以及每个角点的特征向量（embedding vector），使得相同目标的两个角点的特征向量间的距离更小。为了生成更准确的边界框，网络也会预测偏移，来轻微的调整角点的位置。图3是CornerNet的框图。使用hourglass作为骨干网络。骨干网络之后接两个预测模块。一个预测模块用于预测左上角角点，另一个预测右下角角点。每个模块都有相应的角点池化模层，用于对hourglass的特征进行池化，之后得到预测热图、特征向量和预测偏移。本文不使用不同尺度的特征，只使用hourglass输出特征。

![3](/assets/post/2021-08-24-CornerNet/3overview.png)
_图3 overview of cornernet_

### P2.2 角点检测

本文预测2组热图，一组用于左上角角点，一组用于右下角角点。每组热图有C个通道，且无背景通道，热图宽高为H\*W，C为目标检测类别的个数。每个通道是二值掩膜，代表当前位置是某个类别的角点。

每个角点有一个GT位置作为正位置，所有其他位置都为负位置。在训练阶段不是等价的惩罚负样本位置，而是减少了对正位置半径内负位置的惩罚。原因是如果一对错误的角点靠近各自的GT位置，仍然可以产生一个与GT框充分重叠的框，如图4所示。本文根据对象的大小确定半径，确保半径内的一对点对应的框和GT框具有至少t IoU（实验中t=0.3）。得到半径后，通过未归一化的2D高斯核
$${ {e}^{-\frac{ { {x}^{2}}+{ {y}^{2}}}{2{ {\sigma }^{2}}}}}$$
 来进行惩罚，高斯核的中心为GT位置，
 $$\sigma $$
 为半径的1/3。

![4](/assets/post/2021-08-24-CornerNet/4gt.png)
_图4_

令
$${ {p}_{cij}}$$
为预测热图中类别c、位置
$$\left( i,j \right)$$
处的得分，
$${ {y}_{cij}}$$
为通过未归一化的高斯核得到的GT热图，本文设计focal loss的变种：

$${ {L}_{\det }}=\frac{-1}{N}\sum\limits_{c=1}^{C}{\sum\limits_{i=1}^{H}{\sum\limits_{j=1}^{W}{\left\{ \begin{matrix}
   { {\left( 1-{ {p}_{cij}} \right)}^{\alpha }}\log \left( { {p}_{cij}} \right) & if\text{ }{ {y}_{cij}}=1  \\
   { {\left( 1-{ {y}_{cij}} \right)}^{\beta }}{ {\left( { {p}_{cij}} \right)}^{\alpha }}\log \left( 1-{ {p}_{cij}} \right) & otherwise  \\
\end{matrix} \right.}}} \tag{1}$$

其中N为图像中目标的数量，
$$\alpha $$
和
$$\beta $$
为控制每个点分布的超参（实验中设置
$$\alpha =2$$
，
$$\beta =4$$
）。

$$1-{ {y}_{cij}}$$
能够降低GT位置附近的惩罚。

很多网络使用下采样获取全局信息，同时降低显存需求，但会导致网络输出尺寸小于图像尺寸。因而图像上
$$\left( x,y \right)$$
位置会映射到特征图上的
$$\left( \left\lfloor \frac{x}{n} \right\rfloor ,\left\lfloor \frac{y}{n} \right\rfloor  \right)$$
，其中n为下采样率。当从特征图重新映射到输入图像时，会导致精度损失，从而严重影响小边界框和他们GT框的IoU。为了解决这个问题，本文预测位置偏移，在将位置映射回输入尺寸之前轻微调整角点的位置。

$${ {o}_{k}}=\left( \frac{ { {x}_{k}}}{n}-\left\lfloor \frac{ { {x}_{k}}}{n} \right\rfloor ,\frac{ { {y}_{k}}}{n}-\left\lfloor \frac{ { {y}_{k}}}{n} \right\rfloor  \right) \tag{2}$$

其中
$${ {o}_{k}}$$
为偏移，
$${ {x}_{k}}$$
和
$${ {y}_{k}}$$
为角点k的x和y坐标。实际中会预测所有类别共享的左上角偏移，及所有类别共享的右下角偏移。使用smooth L1 loss训练偏移：

$${ {L}_{off}}=\frac{1}{N}\sum\limits_{k=1}^{N}{\text{SmoothL1Loss}\left( { {o}_{k}},{ { {\hat{o}}}_{k}} \right)} \tag{3}$$

smooth L1 loss如下<https://arxiv.org/abs/1504.08083>：

$$\text{smoot}{ {\text{h}}_{\text{L1}}}\left( x \right)=\left\{ \begin{matrix}
   0.5{ {x}^{2}} & if\text{ }\left| x \right|<1  \\
   \left| x \right|-0.5 & otherwise  \\
\end{matrix} \right. \tag{3.1}$$ 

说明：公式3.1中x等于公式3中
$${ {o}_{k}}-{ {\hat{o}}_{k}}$$


### P2.3 角点分组

一张图像中可能出现多个类别的目标，可能检测到多个左上角和右下角角点，从而需要确定一组左上角和右下角角点属于同一个边界框。本文受<https://arxiv.org/abs/1611.05424>中多人姿态估计算法启发。该文检测所有人的关节点，并生成每个关节点的特征向量（embedding），然后根据特征向量间的距离对关节点分组。该算法也能用于本文中。网络对每个检测到的角点预测特征向量，该向量代表预测到的左上角角点和右下角角点是否来自同一个框（相同目标），同一个框的embedding向量的距离更小。之后根据左上角和右下角角点的特征向量的距离对角点分组。特征向量的实际值不重要，因为使用的是特征之间的距离来对角点分组。

本文使用1维的特征向量。令
$${ {e}_{ { {t}_{k}}}}$$
为目标k左上角角点的特征，
$${ {e}_{ { {b}_{k}}}}$$
为目标k右下角角点的特征，本文使用pull loss训练网络来聚集角点，使用push loss来分散角点：

$${ {L}_{pull}}=\frac{1}{N}\sum\limits_{k=1}^{N}{\left[ { {\left( { {e}_{ { {t}_{k}}}}-{ {e}_{k}} \right)}^{2}}+{ {\left( { {e}_{ { {b}_{k}}}}-{ {e}_{k}} \right)}^{2}} \right]} \tag{4}$$

$${ {L}_{push}}=\frac{1}{N\left( N-1 \right)}\sum\limits_{k=1}^{N}{\sum\limits_{\begin{smallmatrix} 
 j=1 \\ 
 j\ne k 
\end{smallmatrix}}^{N}{\max \left( 0,\Delta -\left| { {e}_{k}}-{ {e}_{j}} \right| \right)}} \tag{5}$$

其中
$${ {e}_{k}}$$
为
$${ {e}_{ { {t}_{k}}}}$$
和
$${ {e}_{ { {b}_{k}}}}$$
的平均值，即中心，
$$\Delta =1$$
，push loss是左上角和右下角角点的中心之间相互比较。和训练偏移的损失类似，本文只对GT位置的角点计算pull loss和push loss。

### P2.4 角点池化（Corner Pooling）

角点通常没有局部视觉信息。因而需要分别需要水平向右看物体的最高边缘、垂直向下看物体的最左侧边缘，才能确定左上角的角点。因而该文提出角点池化（Corner Pooling），来给角点编码先验知识，更好的定位角点。

假定需要确定位置
$$\left( i,j \right)$$
的像素是否是左上角点。令
$${ {f}_{t}}$$
和
$${ {f}_{l}}$$
分别为左上角点池化层的输入，
$${ {f}_{ { {t}_{ij}}}}$$
和
$${ {f}_{ { {l}_{ij}}}}$$
分别为
$${ {f}_{t}}$$
和
$${ {f}_{l}}$$
中位置
$$\left( i,j \right)$$
处的向量。对于H\*W的特征图，角点池化层首先最大池化（max-pool）
$${ {f}_{t}}$$
中所有的
$$\left( i,j \right)$$
和
$$\left( i,H \right)$$
的特征到特征向量
$${ {t}_{ij}}$$
中，然后最大池化（max-pool）
$${ {f}_{l}}$$
中所有的
$$\left( i,j \right)$$
和
$$\left( W,j \right)$$
的特征到特征向量
$${ {l}_{ij}}$$
中。最后将
$${ {t}_{ij}}$$
和
$${ {l}_{ij}}$$
的结果相加：

$${ {t}_{ij}}=\left\{ \begin{matrix}
   \max \left( { {f}_{ { {t}_{ij}}}},{ {t}_{\left( i+1 \right)j}} \right) & if\text{ }i<H  \\
   { {f}_{ { {t}_{Hj}}}} & otherwise  \\
\end{matrix} \right. \tag{6}$$

$${ {l}_{ij}}=\left\{ \begin{matrix}
   \max \left( { {f}_{ { {l}_{ij}}}},{ {t}_{i\left( j+1 \right)}} \right) & if\text{ }j<W  \\
   { {f}_{ { {t}_{iW}}}} & otherwise  \\
\end{matrix} \right. \tag{7}$$

公式6中，当前点的值是当前点右侧所有值的最大的。i从H开始\-\-，进行比较。公式7中，当前点的值是当前点下方所有值的最大的。j从W开始\-\-，进行比较。
$${ {t}_{ij}}$$
和
$${ {l}_{ij}}$$
可以使用动态规划高效计算，如图5所示。最终
$${ {t}_{ij}}\text{+}{ {l}_{ij}}$$
得到结果。

![5](/assets/post/2021-08-24-CornerNet/5tlcornerpooling.png)
_图5_

右下角池化层通过类似方式定义。其最大池化
$$\left( 0,j \right)$$
和
$$\left( i,j \right)$$
的特征向量，以及
$$\left( i,0 \right)$$
和
$$\left( i,j \right)$$
的特征向量，并把相应结果相加。角点池化层用在预测模块，来预测热图、特征向量和目标偏移。

预测模块如图6所示。第一部分为修改的残差模块，此处将第一个3\*3卷积替换2个具有128通道的3\*3卷积模块模块，用来处理骨干网络的特征，卷积模块之后为角点池化层。按照残差模块的设计，我们将池化后的特征输入一个具有256个通道的3\*3 Conv-BN层，并添加shortcut支路。修改后的残差模块后面接具有256通道的3\*3卷积模块，3个Conv-ReLU-Conv层，来预测热图、特征向量和目标偏移。

![6](/assets/post/2021-08-24-CornerNet/6predictionmodule.png)
_图6 predictionmodule_

### P2.4 Hourglass网络

CornerNet的骨干网络为hourglass网络。其为全卷积网络，包含至少一个hourglass模块。hourglass模块先通过一系列的conv和max pooling来下采样输入特征，而后通过一系列上采样和卷积层上采样特征到原始分辨率。使用hourglass时，由于上采样丢失了细节信息，因而将输入层和上采样层使用skip层相连。
本文使用的hourglass网络包含2个hourglass模块，但对hourglass模块做了一些修改。我们没有使用max pooling，而是使用stride=2来降低特征图的分辨率。本文降低特征图的分辨率5次，同时特征通道数依次变为(256, 384, 384, 384, 512)。而后使用2个残差模块和一个最近邻上采样模块来上采样特征。每个skip连接也包括2个残差模块。在hourglass模块的中间有4个具有512个通道的残差模块。在hourglass模块之前，使用stride=2、128个通道的7\*7卷积，以及stride=2、256个通道的残差块将图像分辨率降低4倍。

本文训练阶段也使用中间监督。不过我们发现中间预测会降低网络的性能，因而没有将中间预测添加到网络中。我们在第一个hourglass模块的输入和输出都使用1\*1 conv bn，然后使用逐元素相加并加上ReLU和256通道的残差模块，作为第二个hourglass模块的输入。hourglass网络的层数为104。我们仅使用网络最后一层的特征进行预测。

### P2.5 损失

该文使用Adam训练网络。总体的损失为：

$$L={ {L}_{det}}+\alpha { {L}_{pull}}+\beta { {L}_{push}}+\gamma { {L}_{off}} \tag{8}$$

其中
$$\alpha =0.1$$
、
$$\beta =0.1$$
、
$$\gamma =1$$
分别是pull loss，push loss、偏移损失的权重。
$$\alpha $$
和
$$\beta $$
大于等于1时，模型性能变差。


## P3.代码

稍后补上
