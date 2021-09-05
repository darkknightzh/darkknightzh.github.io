---
layout: post
title:  "Weight Normalization（WN）"
date:   2021-08-09 16:00:00 +0800
tags: [deep learning, algorithm, normalization]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/weightnorm>

论文：

<https://arxiv.org/abs/1602.07868>

<br>


**① 权重归一化（Weight Normalization）**

给定k维权重向量w和偏置标量b，以及k维神经元输入向量x，该神经元输出为

$$y=\phi \left( \mathbf{w}\centerdot \mathbf{x}+b \right)$$

其中
$$\phi \left( \centerdot  \right)$$
为激活函数，如ReLU，y为标量输出。

一般使用sgd更新参数**w**和b，为了加速收敛，该文将**w**重新参数化为参数向量**v**和标量g，进而使用sgd训练**v**和g：

$$\mathbf{w}=\frac{g}{\left\| \mathbf{v} \right\|}\mathbf{v}$$

**v**是k维向量，g为标量，$$\left\| \mathbf{v} \right\|$$
为v的欧式模长。这种归一化方式的好处是：确保
$$\left\| \mathbf{w} \right\|=g$$
，独立于参数v。这种重新参数化的方法，称作权重归一化（Weight Normalization，WN）。

之前的权重归一化使用sgd优化**w**，WN则是直接使用sgd优化**v**和g。通过分离权重向量的模长g和权重向量的方向
$$\frac{\mathbf{v}}{\left\| \mathbf{v} \right\|}$$
，可以加快sgd的收敛速度。

可以直接使用g，也可以使用指数的方式
$$g={ {e}^{s}}$$
，其中s为sgd要学习的参数。但实验结果无明显优劣。

**② 梯度计算**

损失L关于**v**和g的梯度如下：

$${ {\nabla }_{g}}L=\frac{ { {\nabla }_{\mathbf{w}}}L\centerdot \mathbf{v}}{\left\| \mathbf{v} \right\|}$$

$${ {\nabla }_{\mathbf{v}}}L=\frac{g}{\left\| \mathbf{v} \right\|}{ {\nabla }_{\mathbf{w}}}L-\frac{g{ {\nabla }_{g}}L}{ { {\left\| \mathbf{v} \right\|}^{2}}}\mathbf{v}$$

其中
$${ {\nabla }_{\mathbf{w}}}L$$
为通常使用的损失关于w的梯度。
$${ {\nabla }_{\mathbf{v}}}L$$
可以写成如下形式：

$${ {\nabla }_{\mathbf{v}}}L=\frac{g}{\left\| \mathbf{v} \right\|}{ {M}_{\mathbf{w}}}{ {\nabla }_{\mathbf{w}}}L \quad \text{with} \quad { {M}_{\mathbf{w}}}=I-\frac{\mathbf{ww}'}{ { {\left\| \mathbf{w} \right\|}^{2}}}$$

$${ {M}_{\mathbf{w}}}$$
是投影矩阵，将计算投影到**w**向量上。因而权重归一化完成了2件事：缩放权重的梯度（缩放系数为
$$\frac{g}{\left\| \mathbf{v} \right\|}$$
），将梯度从当前权重向量**v**投影到**w**。

使用权重归一化时，增加模长
$$\left\| \mathbf{v} \right\|$$
能使得训练对学习率更稳健：如果学习率太大， 未归一化的权重的模长会快速增加，直到达到恰当的学习率。因而使用权重归一化的神经网络能使用更大范围的学习率。另外使用BN的神经网络也有这个特点，也能通过这个分析解释。

**③ 参数初始化**

BN能固定每层特征的幅度。使得优化相比于每层幅度变化太大的网络更易优化。权重归一化没有这个特点，因而初始化参数很重要。**v**服从0均值，0.05标准差的正态分布。初始化b和g时，先使用一个小的batch，通过公式1计算t和y，而后得到当前小batch上的t的均值和标准差
$$\mu \left[ t \right]$$
和
$$\sigma \left[ t \right]$$
，之后使用公式2初始化b和g。

$$t=\frac{\mathbf{v}\centerdot \mathbf{x}}{\left\| \mathbf{v} \right\|}\text{ ,          }y=\phi \left( \frac{t-\mu \left[ t \right]}{\sigma \left[ t \right]} \right) \tag{1}$$

$$g\leftarrow \frac{1}{\sigma \left[ t \right]}\text{ ,          }b\leftarrow \frac{-\mu \left[ t \right]}{\sigma \left[ t \right]} \tag{2}$$

**④ Mean-only Batch Normalization**

权重归一化能确保神经元的尺度和**v**近似无关。但是神经元的均值仍旧依赖于**v**。因而本文结合权重归一化和BN的特殊版本，提出mean-only batch normalization：减去当前batch的均值，但是不除以当前batch的标准差。

$$t=\mathbf{w}\centerdot \mathbf{x}\text{ ,} \quad \quad \tilde{t}=t-\mu \left[ t \right]+b\text{ ,} \quad \quad y= \phi \left( {\tilde{t}} \right)$$

训练时，使用滑动平均的方式更新
$$\mu \left[ t \right]$$
，在测试的时候直接使用
$$\mu \left[ t \right]$$
。
