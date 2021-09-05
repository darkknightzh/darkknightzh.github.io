---
layout: post
title:  "Switchable Normalization（SN）"
date:   2021-08-09 16:00:00 +0800
tags: [deep learning, algorithm, normalization]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/switchablenorm>

<br>

论文：

<https://arxiv.org/abs/1806.10779>

官方pytorch代码：

<https://github.com/ascenoputing/Switchable-Normalization>

作者之一关于SN的中文解读：

<https://zhuanlan.zhihu.com/p/39296570>

<br>

该文提出Switchable Normalization，使用加权BN、IN、LN的归一化方法，其中权重是通过训练得到。

**① 归一化的通用形式**

假设4D张量h为归一化层的输入特征，尺寸为(N,C,H,W)，其中N为样本数量，通道数量，特征高，特征宽，如下图所示。令
$${ {h}_{ncij}}$$
和
$${ {\hat{h}}_{ncij}}$$
为归一化前后的特征，其中
$$n\in \left[ 1,N \right]$$
，
$$c\in \left[ 1,C \right]$$
，
$$i\in \left[ 1,H \right]$$
，
$$j\in \left[ 1,W \right]$$
。令
$$\mu $$
和
$$\sigma $$
为均值和标准差。则：

$${ {\hat{h}}_{ncij}}=\gamma \frac{ { {h}_{ncij}}-\mu }{\sqrt{ { {\sigma }^{2}}+\varepsilon }}+\beta \tag{1}$$

其中
$$\gamma $$
和
$$\beta $$
分别为缩放和平移参数，
$$\varepsilon $$
是保持数值稳定性的小常数。公式(1)表明每个特征都通过
$$\mu $$
和
$$\sigma $$
进行了归一化，且通过
$$\gamma $$
和
$$\beta $$
进行了平移和缩放。
 
![1](/assets/post/2021-08-09-switchablenorm/1switchablenorm.png)
_图1_

IN、LN、BN都共享公式(1)，只不过他们使用不同的特征估计
$$\mu $$
和
$$\sigma $$
：

$${ {\mu }_{k}}=\frac{1}{\left| { {I}_{k}} \right|}\sum\limits_{(n,c,i,j)\in { {I}_{k}}}{ { {h}_{ncij}}}\text{  , } \quad \quad \sigma _{k}^{2}=\frac{1}{\left| { {I}_{k}} \right|}\sum\limits_{(n,c,i,j)\in { {I}_{k}}}{ { {\left( { {h}_{ncij}}-{ {\mu }_{k}} \right)}^{2}}} \tag{2}$$

其中
$$k\in \left\{ in,ln,bn \right\}$$
，用来区分不同的归一化方法。
$${ {I}_{k}}$$
是特征集，
$$\left| { {I}_{k}} \right|$$
是特征的数量。具体而言，
$${ {I}_{in}}$$
，
$${ {I}_{ln}}$$
，
$${ {I}_{bn}}$$
分别为不同方法统计特征的集合。

**IN在图像风格转换中提出**，
$${ {\mu }_{in}},\sigma _{in}^{2}\in { {\mathbb{R}}^{N\times C}}$$
，
$${ {I}_{in}}=\left\{ \left( i,j \right)\left| i\in \left[ 1,H \right],j\in \left[ 1,W \right] \right. \right\}$$
，表明IN有2NC的统计参数，其中每个均值和方差都沿着每个样本的每个通道的(H,W)计算。

**LN在RNN的优化问题中提出**，
$${ {\mu }_{\ln }},\sigma _{ln}^{2}\in { {\mathbb{R}}^{N\times 1}}$$
，
$${ {I}_{ln}}=\left\{ \left( c,i,j \right)\left| c\in \left[ 1,C \right],i\in \left[ 1,H \right],j\in \left[ 1,W \right] \right. \right\}$$
，表明LN有2N的统计参数，其中每个均值和方差都沿着每个样本的(C,H,W)计算。

**BN在CNN的图像分类问题中提出**，用来归一化隐含层的特征图，
$${ {\mu }_{bn}},\sigma _{bn}^{2}\in { {\mathbb{R}}^{C\times 1}}$$
，
$${ {I}_{bn}}=\left\{ \left( n,i,j \right)\left| n\in \left[ 1,N \right],i\in \left[ 1,H \right],j\in \left[ 1,W \right] \right. \right\}$$
，表明LN有2C的统计参数，BN和IN一样，认为每个通道相互独立，但BN的每个均值和方差都沿着每个通道的(N,H,W)计算。

**② SN**

SN形式如下：

$${ {\hat{h}}_{ncij}}=\gamma \frac{ { {h}_{ncij}}-\sum\nolimits_{k\in \Omega }{ { {w}_{k}}{ {\mu }_{k}}}}{\sqrt{\sum\nolimits_{k\in \Omega }{w_{k}^{'}}\sigma _{k}^{2}+\varepsilon }}+\beta \tag{3} $$

其中
$$\Omega $$
是通过不同方法得到的统计特征的集合。本文中
$$\Omega \text{=}\left\{ in,ln,bn \right\}$$
。
$${ {\mu }_{k}}$$
和
$$\sigma _{k}^{2}$$
依旧通过公式(2)计算。但这样会有很大的冗余计算。实际上，LN和BN可以通过IN得到：

$$\begin{align}
  & { {\mu }_{in}}=\frac{1}{HW}\sum\limits_{i,j}^{H,W}{ { {h}_{ncij}}}\text{  , } \quad \quad \sigma _{in}^{2}=\frac{1}{HW}\sum\limits_{i,j}^{H,W}{ { {\left( { {h}_{ncij}}-{ {\mu }_{in}} \right)}^{2}}} \\ 
 & { {\mu }_{\ln }}=\frac{1}{C}\sum\limits_{c=1}^{C}{ { {\mu }_{in}}}\text{  ,} \quad \quad \sigma _{ln}^{2}=\frac{1}{C}\sum\limits_{c=1}^{C}{\left( \sigma _{in}^{2}+\mu _{in}^{2} \right)}-\mu _{ln}^{2} \\ 
 & { {\mu }_{bn}}=\frac{1}{N}\sum\limits_{n=1}^{N}{ { {\mu }_{in}}}\text{  ,} \quad \quad \sigma _{bn}^{2}=\frac{1}{N}\sum\limits_{n=1}^{N}{\left( \sigma _{in}^{2}+\mu _{in}^{2} \right)}-\mu _{bn}^{2} \\ 
\end{align} \tag{4}$$

使用公式(4)，SN的计算复杂度为O(NCHW)，和之前的工作类似。

$${ {w}_{k}}$$
和
$$w_{k}^{'}$$
用来对均值和方差进行加权平均，是很重要的比率。每个
$${ {w}_{k}}$$
和
$$w_{k}^{'}$$
都是一个标量，被所有通道共享。SN中一共有3\*2=6个重要的权重。由于
$$\sum\nolimits_{k\in \Omega }{ { {w}_{k}}}=1$$
，
$$\sum\nolimits_{k\in \Omega }{w_{k}^{'}}=1$$
，进而定义：

$${ {w}_{k}}=\frac{ { {e}^{ { {\lambda }_{k}}}}}{\sum\nolimits_{z\in \left\{ in,ln,bn \right\}}{ { {e}^{ { {\lambda }_{z}}}}}}\text{   }\text{          }k\in \left\{ in,ln,bn \right\}$$

$${ {w}_{k}}$$
通过对
$${ {\lambda }_{in}}$$
，
$${ {\lambda }_{ln}}$$
，
$${ {\lambda }_{bn}}$$
进行softmax得到，
$${ {\lambda }_{in}}$$
，
$${ {\lambda }_{ln}}$$
，
$${ {\lambda }_{bn}}$$
为通过BP学习的控制参数。
$$w_{k}^{'}$$
同理，可学习到
$$\lambda _{in}^{'}$$
，
$$\lambda _{ln}^{'}$$
，
$$\lambda _{bn}^{'}$$
。

**③ 训练阶段**

令
$$\Theta $$
代表网络的参数（如滤波器），
$$\Phi $$
代表控制参数。在SN中，
$$\Phi =\left\{ { {\lambda }_{in}},{ {\lambda }_{ln}},{ {\lambda }_{bn}},\lambda _{in}^{'},\lambda _{ln}^{'},\lambda _{bn}^{'} \right\}$$
。训练使用SN的网络即最小化损失函数
$$L\left( \Theta ,\Phi  \right)$$
，其中
$$\Theta $$
和
$$\Phi $$
使用BP联合优化。

**④ 测试阶段**

测试阶段（由于SN包括IN、LN、BN），IN和LN是每个样本单独统计，BN则是直接使用batch的平均结果，而未使用训练阶段的滑动平均结果。具体包含2个阶段：①冻结网络和所有SN层的参数，并从训练集随机选一些批次的图片，送入网络；②将这些批次中相应SN层的均值和方差进行平均。得到的均值和方差用于SN中的BN。

实验结果表明batch平均比滑动平均收敛速度要快
