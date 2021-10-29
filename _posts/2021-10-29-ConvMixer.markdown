---
layout: post
title:  "Patches Are All You Need?"
date:   2021-10-28 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/ConvMixer>


论文：

<https://openreview.net/pdf?id=TVHS5Y4dNvM>

官方pytorch代码：

<https://github.com/tmp-iclr/convmixer>

<br>


# P1. 摘要

ViT需要使用patch embeddings，将图像的小区域组合成特征，这样方便应用到大图像上。而该文探讨了如下问题：ViT的强大性能是否更多地来自基于patch的表示，而不是来自Transformer结构本身？该文给出的结果是：是的。并且提出了ConvMixer，直接将图像块作为输入，计算空间维度特征和通道维度特征（串联），并在整个网络中保持相同的大小和分辨率。


# P2. 网络结构

ConvMixer网络结构如下图所示。

① 输入c\*n\*n的图像通过patch embedding（conv）被分成子块的特征，而后通过GELU和BN，作为传统网络结构最开始的3层（conv+BN+ReLU），得到特征。

② 特征通过depth个ConvMixer layer（串联），得到新的特征，此时特征分辨率不变，为FCN网络。每个ConvMixer都是残差连接的DWC（通道数不变的分组卷积）+GELU+BN，加上PWC（通道数不变的1\*1卷积）+GELU+BN。

③ 通过global average pooling将特征宽高降低为1，并通过FC层得到分类结果。

![1](/assets/post/2021-10-29-ConvMixer/1.png)

# P3. ConvMixer

该文提出的ConvMixer包含patch embedding layer、重复多次的FCN块（ConvMixer block）。FCN不改变patch embedding得到特征的CHW，如上图所示。patch size=p，embedding 维度=h的patch embedding 可通过
$${ {c}_{in}}$$
输入通道，h输出通道，卷积核大小=p，stride=p的卷积实现：

$${ {z}_{0}}=BN\left( \sigma \left\{ con{ {v}_{ { {c}_{in}}\to h}}\left( X,stride=p,\ker nel\_size=p \right) \right\} \right)$$

ConvMixer block包含depthwise convolution (groups=通道数的分组卷积) 和pointwise convolution（kernel size 1*1）。convMixer在depthwise convolution的特别大的卷积核时工作得最好。每个卷积之后都有激活函数和BN层：

$$z_{l}^{'}=BN\left( \sigma \left\{ ConvDepthwise\left( { {z}_{l-1}} \right) \right\} \right)+{ {z}_{l-1}}$$

$${ {z}_{l+1}}=BN\left( \sigma \left\{ ConvPointwise\left( z_{l}^{'} \right) \right\} \right)$$

多次重复这种结构之后，使用global pooling得到长度为h的特征向量，并送入softmax分类器。


# P4. 代码

convmixer-main的convmixer.py。这边如果在convmixer-main目录下打开vs code，修改train.py参数后，直接调试时，需要将pytorch-image-models/timm/models/convmixer.py中sys.path.append('../../../')改为绝对路径。否则from convmixer import ConvMixer会失败。

代码如下：

```python
class Residual(nn.Module):   # 对应论文图中的残差层
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),   # ConvMixer layer之前的层
        nn.GELU(),
        nn.BatchNorm2d(dim),    # 得到[bs, dim, ing_h/patch_size, ing_w/patch_size]的特征。此处stride=patch_size实现了论文中对图像分块的操作。每个块得到一个单值的特征
        *[nn.Sequential(   # 内部为一个ConvMixer layer
                Residual(nn.Sequential(   # 残差层
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),       # x + (conv(x) + GELU(x) + BN(x))，和论文中图2 ConvMixer layer前三层结构一致
                nn.Conv2d(dim, dim, kernel_size=1),   # 1*1卷积
                nn.GELU(),
                nn.BatchNorm2d(dim)   # 1*1conv(x) + GELU(x) + BN(x)，和论文中图2 ConvMixer layer后三层结构一致
        ) for i in range(depth)],     # depth个ConvMixer layer，和论文中图2的depth个ConvMixer layer一致
        nn.AdaptiveAvgPool2d((1,1)),    # 参数为输出特征的大小。即将输入尺寸转换到1*1的输出特征。得到[bs, dim, 1, 1]的特征
        nn.Flatten(),   # 变成2维。得到[bs, dim]的特征
        nn.Linear(dim, n_classes)   # 分类层。得到[bs, n_classes]的特征
    )
```

代码和上图是对应的。需要注意的是，开始使用nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)来得到patch embedding，实际上是通过stride实现了对图像的分块操作，每个块通过kernel_size=patch_size得到一个单值的特征。

由于使用的是7*7的卷积，因而计算量并不太小。
