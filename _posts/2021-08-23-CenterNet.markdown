---
layout: post
title:  "CenterNet Objects as Points(代码未添加)"
date:   2021-08-23 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/CenterNet>

论文：

<https://arxiv.org/abs/1904.07850>

官方代码：

<https://github.com/xingyizhou/CenterNet>


## P1. 摘要
CenterNet为anchor-free的方法。如图1所示，本文将目标建模为其边界框的中心点。边界框的大小和其他属性可以通过目标中心关键点的特征推断。


![1](/assets/post/2021-08-23-CenterNet/1centernet.png)
_图1 centernet_

本文算法可以认为是一个单形不可知的anchor（single shape-agnostic anchor），如图2所示。优点：① CenterNet基于位置，而不是框的重叠。本文对于前景和背景分类，无需手动设定阈值。② 对于每个目标，本文只有一个正的“anchor”，因而不需要NMS。在特征图上直接提取局部峰值。③ CenterNet使用更大的输出分辨率（stride=4）。

![2](/assets/post/2021-08-23-CenterNet/2centerpixel.png)
_图2_


## P2. 预备知识

假定
$$I\in { {R}^{W\times H\times 3}}$$
为W*H的彩色图像。本文目标是得到关键点热图
$$\hat{Y}\in { {\left[ 0,1 \right]}^{\frac{W}{R}\times \frac{H}{R}\times C}}$$
，其中R为输入相比于输出的步长（stride）。如果是人体姿态估计，则C=17；如果是目标检测，则C=80（COCO数据库）。本文使用默认步长R=4，即输出分辨率为输入分辨率的1/4。预测的
$${ {\hat{Y}}_{x,y,c}}=1$$
为检测的关键点，
$${ {\hat{Y}}_{x,y,c}}=0$$
为背景。本文使用不同的网络来预测图像对应的
$${ {\hat{Y}}_{x,y,c}}$$
：堆叠hourglass网络（stacked hourglass network），ResNet，deep layer aggregation (DLA)。
对于每个类别为c的GT关键点
$$p\in { {R}^{2}}$$
，计算低分辨率的特征
$$\tilde{p}\in \left\lfloor \frac{p}{R} \right\rfloor $$
。

而后将GT关键点使用高斯核映
$${ {Y}_{xyc}}\in \exp \left( -\frac{ { {\left( x-{ { {\tilde{p}}}_{x}} \right)}^{2}}+{ {\left( y-{ { {\tilde{p}}}_{y}} \right)}^{2}}}{2\sigma _{p}^{2}} \right)$$
射到热图
$$Y\in { {\left[ 0,1 \right]}^{\frac{W}{R}\times \frac{H}{R}\times C}}$$
上，其中
$${ {\sigma }_{p}}$$
为目标尺寸自适应的标准差。若相同类别的的两个高斯核重叠，则使用两个高斯核中的更大值。训练时的损失函数为使用focal loss的逻辑回归：

$${ {L}_{k}}=\frac{-1}{N}\sum\limits_{xyc}{\left\{ \begin{matrix}
   { {\left( 1-{ { {\hat{Y}}}_{xyc}} \right)}^{\alpha }}\log \left( { { {\hat{Y}}}_{xyc}} \right) & if\text{ }{ {Y}_{xyc}}=1  \\
   { {\left( 1-{ {Y}_{xyc}} \right)}^{\beta }}\log { {\left( { { {\hat{Y}}}_{xyc}} \right)}^{\alpha }}\log \left( 1-{ { {\hat{Y}}}_{xyc}} \right) & otherwise  \\
\end{matrix} \right.} \tag{1}$$

其中
$$\alpha $$
和
$$\beta $$
为focal loss中的超参。N为图像I中的关键点数量。除以N是为了使所有正focal loss的实例归一化到1。本文使用
$$\alpha =2$$
，
$$\beta \text{=}4$$。

**说明**：公式1中，第一行，
$${ {Y}_{xyc}}$$
为1时：当预测值
$${ {\hat{Y}}_{xyc}}$$
接近1，第一项给与小的惩罚；否则给与大的惩罚。第二行，
$${ {Y}_{xyc}}$$
不为1时：①若
$${ {Y}_{xyc}}$$
在中心点周围，理论上
$${ {\hat{Y}}_{xyc}}$$
为0，实际上
$${ {\hat{Y}}_{xyc}}$$
接近1时，第二项给与大的惩罚，由于距离中心点较近，
$${ {\hat{Y}}_{xyc}}$$
接近1有可能，因而使用第一项降低一下惩罚；②若
$${ {Y}_{xyc}}$$
远离中心点，理论上
$${ {\hat{Y}}_{xyc}}$$
为0，实际上
$${ {\hat{Y}}_{xyc}}$$
接近1时，第二项给与大的惩罚，第一项保证距离中心越远的点的损失的权重越大，保证负样本检测。

由于对输出取整使用离散化，会所带来预测误差，论文预测每个中心点的局部偏移，使用L1损失，偏移和类别无关，即只预测中心点的偏移，而不管是哪个类别的中心点，这样可以降低输出通道个数。L1损失如下：

$${ {L}_{off}}=\frac{1}{N}\sum\limits_{p}{\left| { { {\hat{O}}}_{ {\tilde{p}}}}-\left( \frac{p}{R}-\tilde{p} \right) \right|} \tag{2}$$

其中
$${ {\hat{O}}_{ {\tilde{p}}}}$$
为中心点预测坐标偏移，
$$\left( \frac{p}{R}-\tilde{p} \right)$$
为中心点实际坐标偏移。该L1损失只考虑关键点位置
$$\tilde{p}$$
，而不考虑其他位置。

**说明**：

① 不考虑具体类别时，偏移只考虑目标的x和y坐标，参数量：2\*目标个数；

② 考虑具体类别时，偏移需要考虑每个类别的目标的x和y坐标，参数量：2\*目标个数\*分类类别。


## P3. Objects as Points

假定
$$\left( x_{1}^{\left( k \right)},y_{1}^{\left( k \right)},x_{2}^{\left( k \right)},y_{2}^{\left( k \right)} \right)$$
为类别为
$${ {c}_{k}}$$
的目标k的边界框。其中心为
$$\left( \frac{x_{1}^{\left( k \right)}+x_{2}^{\left( k \right)}}{2},\frac{y_{1}^{\left( k \right)}+y_{2}^{\left( k \right)}}{2} \right)$$
，本文使用关键点预测器
$$\hat{Y}$$
来预测所有的中心点。另外，对每个目标k，还拟合目标大小
$${ {s}_{k}}=\left( x_{2}^{\left( k \right)}-x_{1}^{\left( k \right)},y_{2}^{\left( k \right)}-y_{1}^{\left( k \right)} \right)$$
。为了降低计算负担，对所有目标类别使用单独的预测器
$$\hat{S}\in { {R}^{\frac{W}{R}\times \frac{H}{R}\times 2}}$$
来预测目标宽高：

$${ {L}_{size}}=\frac{1}{N}\sum\limits_{k=1}^{N}{\left| { { {\hat{S}}}_{pk}}-{ {s}_{k}} \right|} \tag{3}$$

本文不缩放宽高，而是直接预测宽高的值。并且给预测宽高的损失较小的权重
$${ {\lambda }_{size}}$$
。总体的损失函数为：

$${ {L}_{\det }}\text{=}{ {L}_{k}}+{ {\lambda }_{size}}{ {L}_{size}}+{ {\lambda }_{off}}{ {L}_{off}} \tag{4}$$

其中
$${ {\lambda }_{size}}\text{=}0.1$$
，
$${ {\lambda }_{off}}\text{=}0.01$$
。本文使用一个网络来预测关键点
$$\hat{Y}$$
（目标中心），目标中心偏移
$$\hat{O}$$
和目标宽高
$$\hat{S}$$
。因而每个位置网络有C+4个输出。所有输出共享相同的骨干网络。骨干网络会通过独立的子网络得到每个预测信息，子网络为：3\*3 卷积+ReLU+1\*1 卷积。图3显示了网络输出。

![3](/assets/post/2021-08-23-CenterNet/3output.png)
_图3 output_

**从点到边界框**：推断阶段，首先提取每个类别的峰值（该点值大于等于其周围8个点的值作为峰值），并保留100个分值最高的峰值。令
$${ {\hat{P}}_{c}}$$
为n个检测到的类别为c的中心点集合：
$$\hat{P}=\left\{ \left( { { {\hat{x}}}_{i}},{ { {\hat{y}}}_{i}} \right) \right\}_{i=1}^{n}$$
。每个关键点位置为
$$\left( { {x}_{i}},{ {y}_{i}} \right)$$
。本文使用关键点的值
$${ {\hat{Y}}_{xiyic}}$$
作为其检测的置信度，从而得到检测框

$$\left( { { {\hat{x}}}_{i}}+\delta { { {\hat{x}}}_{i}}-{ { { {\hat{w}}}_{i}}}/{2}\;,\text{ }{ { {\hat{y}}}_{i}}+\delta { { {\hat{y}}}_{i}}-{ { { {\hat{h}}}_{i}}}/{2}\;,\text{ }{ { {\hat{x}}}_{i}}+\delta { { {\hat{x}}}_{i}}+{ { { {\hat{w}}}_{i}}}/{2,\text{ }{ { {\hat{y}}}_{i}}+\delta { { {\hat{y}}}_{i}}+{ { { {\hat{h}}}_{i}}}/{2}\;}\; \right)$$
&nbsp;

其中
$$\left( \delta { { {\hat{x}}}_{i}},\delta { { {\hat{y}}}_{i}} \right)={ {\hat{O}}_{ { { {\hat{x}}}_{i}},{ { {\hat{y}}}_{i}}}}$$
为预测的偏移，
$$\left( { { {\hat{w}}}_{i}},{ { {\hat{h}}}_{i}} \right)={ {\hat{S}}_{ { { {\hat{x}}}_{i}},{ { {\hat{y}}}_{i}}}}$$
为预测的宽高。所有的输出直接通过关键点估计，不需要NMS。关键点峰值提取可以代替NMS，并且可以使用3\*3 max pooling高效得到关键点峰值。

说明：可以将CenterNet应用于3D检测和人体姿态估计。具体请看论文。

**网络结构**：如图4所示

![4](/assets/post/2021-08-23-CenterNet/4model.png)
_图4 model diagrams_

## P4.代码

之后补全
