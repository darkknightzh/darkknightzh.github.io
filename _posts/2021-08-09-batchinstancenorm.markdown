---
layout: post
title:  "Batch-Instance Normalization（BIN）"
date:   2021-08-09 16:00:00 +0800
tags: [deep learning, algorithm, normalization]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/batchinstancenorm>


论文：

<https://arxiv.org/abs/1805.07925>

Instance normalization的缺点是其完全抹去了风格信息(it completely erases style information)，其优点是能应用在风格变换（style transfer）和图像-图像变换（image-to-image translation）的场景，但是在对比度很重要的情况下（如天气分类，天空的亮度很重要），其可能会有问题。Batch-instance normalization通过学习每个通道需要使用多少的风格信息，来解决这个问题。

该文假设卷积特征的均值和方差能代表其风格属性。换句话说，每个特征图的信息可被分成两部分：风格（激活函数的均值和方差）+形状（激活函数的空间结构）。从这个角度来说，IN可以认为是在保留形状信息的前提下，归一化每个特征图的风格信息。IN虽然可以减少不需要的风格变化，但在任务中风格信息携带主要特征时，IN会导致信息丢失。BN归一化batch中的特征响应，在batch size不是太小时，能够保留实例级别的风格变化信息。然而在风格不一致时会导致问题复杂化，其缺乏解决该问题的能力。即可认为特征图包含风格信息和形状信息。IN能保留形状信息，但会丢失风格信息（风格变换，变换到其他风格，因而丢失了当前风格信息）。BN会保留风格信息，但会丢失形状信息。

令
$$\mathbf{x}\in { {\mathbb{R}}^{N\times C\times H\times W}}$$
为当前层的输入，
$${ {x}_{nchw}}$$
为第nchw个元素，其中h和w为空间位置，c为通道索引，n为在当前batch中的索引。

BN使用当前batch中每个通道的均值和方差归一化每个通道的特征：

$$\hat{x}_{nchw}^{(B)}=\frac{ { {x}_{nchw}}-\mu _{c}^{(B)}}{\sqrt{\sigma { {_{c}^{2}}^{(B)}}+\varepsilon }}$$

$$\mu _{c}^{(B)}=\frac{1}{NHW}\sum\limits_{N}{\sum\limits_{H}{\sum\limits_{W}{ { {x}_{nchw}}}}}$$

$$\sigma { {_{c}^{2}}^{(B)}}=\frac{1}{NHW}\sum\limits_{N}{\sum\limits_{H}{\sum\limits_{W}{ { {\left( { {x}_{tilm}}-\mu _{c}^{(B)} \right)}^{2}}}}}$$

令$${ {\mathbf{\hat{x}}}^{(B)}}\text{=}\left\{ \hat{x}_{nchw}^{(B)} \right\}$$为BN归一化后的输出特征向量。

IN使用每个实例的特征统计单独归一化当前batch中的每个实例：

$$\hat{x}_{nchw}^{(I)}=\frac{ { {x}_{nchw}}-\mu _{nc}^{(I)}}{\sqrt{\sigma { {_{nc}^{2}}^{(B)}}+\varepsilon }}$$

$$\mu _{nc}^{(I)}=\frac{1}{HW}\sum\limits_{H}{\sum\limits_{W}{ { {x}_{nchw}}}}$$

$$\sigma { {_{nc}^{2}}^{(I)}}=\frac{1}{HW}\sum\limits_{H}{\sum\limits_{W}{ { {\left( { {x}_{tilm}}-\mu _{nc}^{(I)} \right)}^{2}}}}$$

令
$${ {\mathbf{\hat{x}}}^{(I)}}\text{=}\left\{ \hat{x}_{nchw}^{(I)} \right\}$$
为IN归一化后的输出特征向量。

可见在
$${ {\mathbf{\hat{x}}}^{(B)}}$$
中保留了风格的变化，但在
$${ {\mathbf{\hat{x}}}^{(I)}}$$
中丢失了风格的变化（ps：不知道为何保留，又为何丢失。。。）。因而作者引入可学习的参数
$$\rho \in { {\left[ 0,1 \right]}^{C}}$$
，提出Batch-Instance Normalization (BIN)：

$$\mathbf{y}=\left( \rho \centerdot { { {\mathbf{\hat{x}}}}^{(B)}}+\left( 1-\rho  \right)\centerdot { { {\mathbf{\hat{x}}}}^{(I)}} \right)\centerdot \gamma +\beta $$

其中
$$\gamma $$、$$\beta $$
均是C维的向量，代表仿射变换的参数，
$$\mathbf{y}\in { {\mathbb{R}}^{N\times C\times H\times W}}$$
为BIN的输出。
$$\rho $$
是一个C维的向量，每次更新
$$\rho $$
时，都将其截断到[0, 1]之间，确保每个值都在[0, 1]之间：

$$\rho \leftarrow cli{ {p}_{\left[ 0,1 \right]}}\left( \rho -\eta \Delta \rho  \right)$$

$$\rho $$
可解释为门控向量，来决定是保持还是舍弃每个通道的风格变化。风格信息为主时，
$$\rho $$
趋近1时，BN权重大；形状信息为主时，
$$\rho $$
趋近于0，IN权重大。

实验发现，为了更新
$$\rho $$
，最好增大学习率，和理论相符。理论上损失
$$l$$
关于
$$\rho $$
的梯度需要乘以
$${ {\mathbf{\hat{x}}}^{(B)}}$$
和
$${ {\mathbf{\hat{x}}}^{(I)}}$$
的差值（如下式），当前batch中的风格变化不重要时，差值会比较小，因而损失
$$l$$
关于
$$\rho $$
的梯度往往比较小。为了加快训练
$$\rho $$
，需要增大学习率。

$$\frac{\partial l}{\partial { {\rho }_{c}}}\text{=}{ {\gamma }_{c}}\sum\limits_{N}{\sum\limits_{H}{\sum\limits_{W}{\left( \mathbf{\hat{x}}_{nchw}^{(B)}-\mathbf{\hat{x}}_{nchw}^{(I)} \right)\frac{\partial l}{\partial { {y}_{nchw}}}}}}$$

其中
$${ {\rho }_{c}}$$
是
$$\rho $$
的每一个元素（
$$\rho $$
是一个C维的向量）

BIN可应用在目标识别、多领域学习（如混合领域分类、训练和测试域有一定偏移的域适应）、图像风格变换等领域。
