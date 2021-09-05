---
layout: post
title:  "Group Normalization（GN）"
date:   2021-08-09 16:00:00 +0800
tags: [deep learning, algorithm, normalization]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/groupnorm>


论文：

<https://arxiv.org/abs/1803.08494>

说明：
Batch Norm(BN)，Layer Norm(LN)，Instance Norm(IN)，Group Norm(GN)

<br>

通用的特征归一化，包括BN，LN，IN和GN，定义如下：

$${ {\hat{x}}_{i}}=\frac{1}{ { {\sigma }_{i}}}\left( { {x}_{i}}-{ {\mu }_{i}} \right)$$

其中x为输入特征。i为索引，在输入为2D图像是，
$$i=\left( { {i}_{N}},{ {i}_{C}},{ {i}_{H}},{ {i}_{W}} \right)$$
为(N, C, H, W)顺序的4D索引向量，其中N为batch轴，C为通道轴，H和W为空间高度和宽度轴。

$$\mu $$
和
$$\sigma $$
分别为均值和标准差，分别通过下式计算：

$${ {\mu }_{i}}=\frac{1}{m}\sum\limits_{k\in { {S}_{i}}}{ { {x}_{k}}}$$

$${ {\sigma }_{i}}=\sqrt{\frac{1}{m}\sum\limits_{k\in { {S}_{i}}}{ { {\left( { {x}_{k}}-{ {\mu }_{i}} \right)}^{2}}}+\varepsilon }$$

$${ {S}_{i}}$$
为计算均值和标准差需要用到的像素（特征集合），m为该集合的元素数量。

根据这种定义方式，不同的特征归一化的方法的区别就是$${ {S}_{i}}$$的定义方式不同。

**BN**中，
$${ {S}_{i}}$$
定义如下：

$${ {S}_{i}}=\left\{ k|{ {k}_{c}}={ {i}_{c}} \right\}$$

其中
$${ {i}_{c}}$$
和
$${ {k}_{c}}$$
沿着C轴的i和k的子索引。这意味着BN中同一层相同通道索引的特征共同被归一化，即每个通道上，BN沿着(N,H,W)轴计算
$$\mu $$
和
$$\sigma $$
。

**LN**中，
$${ {S}_{i}}$$
定义如下：

$${ {S}_{i}}=\left\{ k|{ {k}_{N}}={ {i}_{N}} \right\}$$

对每个样本，LN沿着(C,H,W)轴计算
$$\mu $$
和
$$\sigma $$
。

**IN**中，
$${ {S}_{i}}$$
定义如下：

$${ {S}_{i}}=\left\{ k|{ {k}_{N}}={ {i}_{N}},{ {k}_{c}}={ {i}_{c}} \right\}$$

对每个样本和每个通道，IN沿着(H, W)轴计算
$$\mu $$
和
$$\sigma $$
。

本文提出的**GN**中，
$${ {S}_{i}}$$
定义如下：

$${ {S}_{i}}=\left\{ k|{ {k}_{N}}={ {i}_{N}},\left\lfloor \frac{ { {k}_{c}}}{C/G} \right\rfloor =\left\lfloor \frac{ { {i}_{c}}}{C/G} \right\rfloor  \right\}$$

其中G为组的数量，默认=32。C/G为每组的通道数量。
$$\left\lfloor \centerdot  \right\rfloor $$
为取整操作，
$$\left\lfloor \frac{ { {k}_{c}}}{C/G} \right\rfloor =\left\lfloor \frac{ { {i}_{c}}}{C/G} \right\rfloor $$
指在每组通道都沿着C轴按顺序存储的前提下，索引i和k在通道中的相同组里面。GN沿着(H,W)轴和C/G进行分组的通道计算
$$\mu $$
和
$$\sigma $$
。

BN，LN，IN，GN通过下式**对每个通道学习线性变换，以补偿可能丧失的表征能力**。

$${ {y}_{i}}=\gamma { {\hat{x}}_{i}}+\beta $$

其中
$$\gamma $$
和
$$\beta $$
为可训练的缩放和平移参数（通过
$${ {i}_{c}}$$
索引，为了简化符号，此处省略）。

**GN和LN的关系**：当组数G=1时，GN成为LN。LN假设每层所有通道贡献相同。但是在卷积层中，这个假设不那么有效。GN的限制比LN少，因为GN假设每组通道（而不是所有通道）共享相同的均值和方差，这样模型仍旧能对每组学习不同的分布。这样GN能比LN有更强的表达能力。

**GN和IN的关系**：当祖师G=C（每组一个通道）时，GN成为IN。但是IN只能在空间维度上计算均值和方差，其无法探索通道间的相关性。

下图展示了Batch Norm(BN)，Layer Norm(LN)，Instance Norm(IN)，Group Norm(GN)的计算差异。简言之，在当前batch的NCHW输入数据中，BN统计每个C维度上NHW的mean和var，LN统计每个N（batch）维度上CHW的mean和var，IN统计每个CN维度上HW的mean和var，GN统计每个N维度上全部HW、部分C的mean和var。

![1](/assets/post/2021-08-09-groupnorm/1norm.png)
_图1_