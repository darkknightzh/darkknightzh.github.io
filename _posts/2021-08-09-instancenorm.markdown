---
layout: post
title:  "Instance Normalization（IN, contrast normalization）"
date:   2021-08-09 16:00:00 +0800
tags: [deep learning, algorithm, normalization]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/instancenorm>


论文：

<https://arxiv.org/abs/1607.08022>

Layer Normalization不依赖batch size，统计同层内所有神经元响应值的均值和方差，Instance Normalization则将统计范围进一步缩小到HW维度。

令
$$x\in { {\mathbb{R}}^{T\times C\times W\times H}}$$
为输入的包含T张图像的张量，其中T为batch size，C为通道数，W为特征宽度，H为特征高度。令
$${ {x}_{tijk}}$$
为
$$x$$
的第tijk个元素，其中k和j为空间维度，i为通道数（图像是RGB时为3通道），t为特征在当前batch中的索引。则对比度归一化的简化版本（a simple version of contrast normalization）为：

$${ {y}_{tijk}}=\frac{ { {x}_{tijk}}}{\sum\nolimits_{l=1}^{W}{\sum\nolimits_{m=1}^{H}{ { {x}_{tilm}}}}}$$

可见对比度归一化只在WH通道上进行，另外，在conv和ReLU之后，如何实现该函数，不是很明确（It is unclear how such as function could be implemented as a sequence of ReLU and convolution operator），此处意思应该是，对比度归一化是建立在图像的基础上，通过conv和ReLU之后已经变成了特征，因而在特征上如何定义对比度归一化，不是很明确。

另一方面，传统的GAN中的生成器使用了BN作为归一化层，BN和上面的contrast normalization的区别是BN将归一化应用到整个batch的图像上：

$${ {y}_{tijk}}=\frac{ { {x}_{tijk}}-{ {\mu }_{i}}}{\sqrt{\sigma _{i}^{2}+\varepsilon }}$$

$${ {\mu }_{i}}=\frac{1}{HWT}\sum\limits_{t=1}^{T}{\sum\limits_{l=1}^{W}{\sum\limits_{m=1}^{H}{ { {x}_{tilm}}}}}$$

$$\sigma _{i}^{2}=\frac{1}{HWT}\sum\limits_{t=1}^{T}{\sum\limits_{l=1}^{W}{\sum\limits_{m=1}^{H}{ { {\left( { {x}_{tilm}}-{ {\mu }_{i}} \right)}^{2}}}}}$$

为了结合BN和特定实例的归一化的效果，作者提出Instance Normalization（IN，也称作contrast normalization）：

$${ {y}_{tijk}}=\frac{ { {x}_{tijk}}-{ {\mu }_{ti}}}{\sqrt{\sigma _{ti}^{2}+\varepsilon }}$$

$${ {\mu }_{ti}}=\frac{1}{HW}\sum\limits_{l=1}^{W}{\sum\limits_{m=1}^{H}{ { {x}_{tilm}}}}$$

$$\sigma _{ti}^{2}=\frac{1}{HW}\sum\limits_{l=1}^{W}{\sum\limits_{m=1}^{H}{ { {\left( { {x}_{tilm}}-{ {\mu }_{ti}} \right)}^{2}}}}$$

作者在文中将生成器g中所有的BN都替换成了IN，IN可以阻止特定实例的均值偏移和方差偏移，从而简化训练过程。另一方面，和BN不同的是，在测试阶段，也使用IN。

由IN的公式可见，IN是统计WH维度上特征的均值和方差，因而IN适用于CNN场景，但是对于RNN或MLP（MLP指全是FC的网络，包含ReLU，但不包含conv），由于其除去BC维度之外，只剩一个单独的神经元，输出也是单值，而非CNN的二维平面，此时统计单值的均值和方差没有意义，因而RNN和MLP无法使用Instance Normalization。

Instance Normalization对于风格转换等图片生成的任务，效果明显优于BN，但在图像分类等场景效果不如BN。
