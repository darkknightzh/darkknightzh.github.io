---
layout: post
title:  "深度学习中的各种卷积"
date:   2021-07-23 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/differentconv>


参考网址：

<https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215>

<https://zhuanlan.zhihu.com/p/257145620>

说明：大部分都是翻译自第一个参考网址

<br>


# P1. 卷积
深度学习中2D卷积如下图所示，Dout个Din\*h\*w的卷积核分别和输入特征进行卷积，得到Dout个Hout\*Wout的特征（每个特征都只有1个通道），再将这些特征特征拼接，得到Dout\*H\*W的输出（如下图a）。由于每个Din\*h\*w的卷积核只在输入特征的H和W方向上滑动（如下图b），因而称作2D卷积。

![1\_1a](/assets/post/2021-07-23-differentconv/1_1a.png)
_（a）_

![1\_1b](/assets/post/2021-07-23-differentconv/1_1b.png)
_（b）_

不考虑batch时，输入特征input尺寸为
$$\left[ { {C}_{in}},{ {H}_{in}},{ {W}_{in}} \right]$$
，卷积核weight尺寸
$$\left[ { {C}_{out}},{ {C}_{in}},{ {H}_{k}},{ {W}_{k}} \right]$$
，偏置bias尺寸
$$\left[ { {C}_{out}} \right]$$
，输出特征out尺寸为
$$\left[ { {C}_{out}},{ {H}_{out}},{ {W}_{out}} \right]$$
，
2D卷积参数量在不考虑bias时为
$${ {C}_{out}}\times { {C}_{in}}\times { {H}_{k}}\times { {W}_{k}}$$
；考虑bias时为
$${ {C}_{out}}\times { {C}_{in}}\times { {H}_{k}}\times { {W}_{k}}+{ {C}_{out}}$$
。

具体可参见下面代码：

```python
from __future__ import print_function
from __future__ import division

import torch.nn as nn
import numpy as np

class testNet(nn.Module):
   def __init__(self):
      super(testNet, self).__init__()
      self.conv1 = nn.Conv3d(in_channels=3, out_channels=10, kernel_size=(6,5,4), stride=1, padding=1, bias=False)

   def forward(self, x):
      x = self.conv1(x)
      return x

def get_total_params(model):
   model_parameters = filter(lambda p: p.requires_grad, model.parameters())
   num_params = sum([np.prod(p.size()) for p in model_parameters])
   return num_params

def main():
   net = testNet()
   print(get_total_params(net))

if __name__ == '__main__':
   main()
```

上面代码3D卷积无bias，此时输出结果为3600

3600=10\*3\*6\*5\*4

有bias时输出结果为3610=10\*3\*6\*5\*4+10

2D卷积的计算量：
$${ {C}_{out}}\times { {C}_{in}}\times { {H}_{k}}\times { {W}_{k}}\times \left( { {H}_{in}}-{ {H}_{k}}+1 \right)\times \left( { {W}_{in}}-{ {W}_{k}}+1 \right)$$

3D卷积输出特征大小为：

$${ {H}_{out}}=\left\lfloor \frac{ { {H}_{in}}+2\times padding[0]-dilation[0]\times (kernel\_size[0]-1)-1}{stride[0]}+1 \right\rfloor $$

$${ {W}_{out}}=\left\lfloor \frac{ { {W}_{in}}+2\times padding[1]-dilation[1]\times (kernel\_size[1]-1)-1}{stride[1]}+1 \right\rfloor $$

2D卷积能在2维空间中编码目标的空间关系。


# P2. 3D卷积

2D卷积是指在3D立方体数据上进行的2D卷积，滤波器通道数Cin和输入特征通道数Cin相同，每个卷积核只在输入特征的高、宽这两个方向上滑动。在3D卷积中，滤波器的通道数小于输入特征的通道数，因而每个卷积核可以在输入特征的通道、高、宽这3个方向上滑动，因而输出为3D数据。如下图。

![2](/assets/post/2021-07-23-differentconv/2.png)

假定3D卷积输入特征
$$\left[ { {C}_{in}},{ {D}_{in}},{ {H}_{in}},{ {W}_{in}} \right]$$
，卷积核尺寸
$$\left[ { {C}_{out}},{ {C}_{in}},{ {D}_{k}},{ {H}_{k}},{ {W}_{k}} \right]$$
，在具体计算时，
$${ {C}_{out}}$$
个
$$\left[ { {C}_{in}},{ {D}_{k}},{ {H}_{k}},{ {W}_{k}} \right]$$
的卷积核和输入特征进行卷积，得到
$${ {C}_{out}}$$
个
$$\left[ { {D}_{out}},{ {H}_{out}},{ {W}_{out}} \right]$$
的特征，拼接后得到
$$\left[ { {C}_{out}},{ {D}_{out}},{ {H}_{out}},{ {W}_{out}} \right]$$
的输出。

3D卷积参数量在不考虑bias时为
$${ {C}_{out}}\times { {C}_{in}}\times { {D}_{k}}\times { {H}_{k}}\times { {W}_{k}}$$
；考虑bias时为
$${ {C}_{out}}\times { {C}_{in}}\times { {D}_{k}}\times { {H}_{k}}\times { {W}_{k}}+{ {C}_{out}}$$
。

3D卷积输出特征大小为：

$${ {D}_{out}}=\left\lfloor \frac{ { {D}_{in}}+2\times padding[0]-dilation[0]\times (kernel\_size[0]-1)-1}{stride[0]}+1 \right\rfloor $$

$${ {H}_{out}}=\left\lfloor \frac{ { {H}_{in}}+2\times padding[1]-dilation[1]\times (kernel\_size[1]-1)-1}{stride[1]}+1 \right\rfloor $$

$${ {W}_{out}}=\left\lfloor \frac{ { {W}_{in}}+2\times padding[2]-dilation[2]\times (kernel\_size[2]-1)-1}{stride[2]}+1 \right\rfloor $$

3D卷积能在3D空间中编码空间关系。这种3D关系在某些应用中很重要，如生物医学成像中的3D分割/重建、CT和MRI等在3D空间中弯曲的血管。


# P3. 转置卷积（反卷积，Transposed Convolution, Deconvolution）

## P3.1. 转置卷积

反卷积为卷积的逆过程，后来不常用反卷积，而常使用转置卷积，用来对输入特征进行上采样，提高输出特征分辨率，增大输出特征感受野。

可以使用卷积实现转置卷积。下图使用3\*3的核，并使用2\*2的padding（填充0），将2\*2的输入上采样成4\*4的输出。

![3\_1](/assets/post/2021-07-23-differentconv/3_1.gif)

将输入特征之间插入不同0之后，可以讲输入映射到不同的输出尺寸，如下图，将2\*2的输入，每个特征之间插入1个0，映射到5\*5的输出（padding仍旧为2\*2）。

![3\_2](/assets/post/2021-07-23-differentconv/3_2.gif)

令C代表卷积的卷积核，large代表输入图像，small代表卷积后的输出图像。通过卷积（矩阵乘法），将大图像large下采样到小图像small：C\*large=small。如下图，将输入平铺为16\*1的矩阵，并将间距和变换为4\*16的稀疏矩阵。之后应用矩阵乘法，得到4\*1的矩阵（4\*16 \* 16\*1=4\*1），并变换回2\*2的输出。

![3\_3](/assets/post/2021-07-23-differentconv/3_3.png)

如果在C\*large=small两边都乘上
$${ {C}^{T}}$$
，可得到
$${ {C}^{T}}*small=large$$
（只有在卷积核C为正交矩阵时，
$${ {C}^{T}}*C=I$$
才成立），如下图所示。

![3\_4](/assets/post/2021-07-23-differentconv/3_4.png)

这就是“转置卷积”的由来。转置卷积的输出分辨率见<https://arxiv.org/abs/1603.07285>中Relationship 13 and Relationship 14。

## P3.2. 棋盘格伪影

使用转置卷积时会出现棋盘格伪影（Checkerboard artifacts），如下图所示。具体可参见论文“Deconvolution and Checkerboard Artifacts”。

![3\_5](/assets/post/2021-07-23-differentconv/3_5.png)

主要原因如下：

棋盘格伪影由转置卷积的不均匀重叠（uneven overlap）引发。这种重叠导致输出特征从输入特征获取到的信息的数量不同（Such overlap puts more of the metaphorical paint in some places than others.）。

下图的顶端为输入，底端为转置卷积的输出。经过转置卷积，小尺寸的输入映射到大尺寸的输出。（a）的stride=1，滤波器大小为2，如红色区域，输入的第一个像素映射到输出的第一个和第二个像素。输入的第一个和第三个像素均会映射到输出的第二个像素。总体上来说，输出的中间部分的像素从输入像素获取相同数量的信息。这里存在卷积核重叠的区域。当卷积核尺寸增大到图（b）中的3时，接收相同数量信息的中心部分会收缩（范围变小）。由于重叠是均匀的，因而这不是大问题。输出特征的中心部分仍旧会从输入特征获取相同数量的信息。

![3\_6](/assets/post/2021-07-23-differentconv/3_6.png)

当stride=2时，下图（a）中滤波器大小=2，输出特征的所有像素从输入获取相同数量的信息（均从输入特征的单个像素获取信息），转置卷积在这种情况下没有重叠现象。当设置滤波器大小=4时，如图（b）所示，均匀重叠区域缩小，但是仍旧能使用输出特征的中心区域作为有效输出，这部分区域内每个像素从输入获取相同数量的信息。当设置滤波器大小为3或5，如图（c）和（d），此时输出特征的每个像素相比其相邻像素会从输入获取不同数量的信息。此时输出没有连续且均匀重叠的区域。

![3\_7](/assets/post/2021-07-23-differentconv/3_7.png)

当滤波器大小不能被stride整除时，转置卷积会有不均匀重叠。不均匀重叠导致输出特征从输入特征获取的信息数量不同，产生了棋盘格伪影。实际上，在二维上两个方向的不均匀相乘，不均匀变成了平方关系，因而不均匀重叠在二维上更加严重。

使用转置卷积时，有2种方式可以减少伪影。一是确保滤波器大小可以被stride整除，来避免不均匀重叠问题。二是可以设置stride=1的转置卷积来减少棋盘格伪影。然而正如许多新的模型中所示，伪影仍旧可能出现。

该文（<https://distill.pub/2016/deconv-checkerboard/>）进一步提出了一种更好的上采样方法来避免棋盘格伪影：首先缩放图像（使用最近邻插值或双线性差值），然后使用传统的卷积。


# P4. 空洞卷积（Dilated Convolution, Atrous Convolution）

空洞卷积见论文

<https://arxiv.org/abs/1412.7062>

<https://arxiv.org/abs/1511.07122>

空洞卷积在卷积核上插入0使得卷积核膨胀，在不增加参数量的情况下，增加模型感受野。pooling的下采样会导致信息丢失，是不可逆的，而空洞卷积可以替代pooling。

标准离散卷积如下：

$$\left( F*k \right)\left( \mathbf{p} \right)=\sum\nolimits_{\mathbf{s}+\mathbf{t}=\mathbf{p}}{F\left( \mathbf{s} \right)k\left( \mathbf{t} \right)}$$

![4\_1](/assets/post/2021-07-23-differentconv/4_1.gif)

空洞卷积如下：

$$\left( F{ {*}_{l}}k \right)\left( \mathbf{p} \right)=\sum\nolimits_{\mathbf{s}+l\mathbf{t}=\mathbf{p}}{F\left( \mathbf{s} \right)k\left( \mathbf{t} \right)}$$

![4\_2](/assets/post/2021-07-23-differentconv/4_2.gif)

当l=1时，空洞卷积和标准卷积相同。

空洞卷积和转置卷积的区别是：转置卷积在图像（或特征图）上插入0，空洞卷积则在卷积核上插入0来使核“膨胀”。参数l（扩张率，dilation rate）表示希望将核加宽的程度。通常在卷积核元素之间插入l-1个0。当l=1,2,4时分别如下图（空洞卷积的好处是可以在不增加额外开销的情况下增大感受野）：

![4\_3](/assets/post/2021-07-23-differentconv/4_3.png)

图中3\*3红点代表经过卷积后，输出图像为3\*3像素。虽然这些空洞卷积的结果都是3\*3像素，但是感受野相差很大。l=1时感受野是3\*3，l=2时感受野是7\*7，l=3时感受野增大到15\*15。与此同时，这三种情况下模型参数相同。因而空洞卷积可以在不增加卷积核大小的情况下，增大输出特征的感受野，在堆叠多个空洞卷积的情况下很有用。

论文“Multi-scale context aggregation by dilated convolutions”中作者使用空洞卷积建立了一个多层网络，其中扩张率l每层呈指数级增加，从而感受野随层呈指数级增加，而参数量仅线性增加。论文中使用空洞卷积在不损失分辨率的情况下聚合多尺度上下文信息（contextual information）。

空洞卷积的缺点是由于kernel的不连续，导致信息的连续性会有损失，对于密集的纹理信息处理不好。


# P5. 扁平卷积（Flattened convolutions）

最先由“Flattened convolutional neural networks for feedforward acceleration”提出，论文为<https://arxiv.org/abs/1412.5474>

该卷积将标准的卷积分解成3个1D卷积，这种思想类似于空间可分卷积（spatial separable convolution），将其中的空间滤波器分为2个秩=1的滤波器。

![5](/assets/post/2021-07-23-differentconv/5.png)

需要注意的是，如果标准卷积核的秩=1，这些核总能分解成3个1D滤波器的叉乘（cross-products）。但是由于标准滤波器的秩总是大于1，因而该假设是一个强条件。正像文中指出的“As the difficulty of classification problem increases, the more number of leading components is required to solve the problem… Learned filters in deep networks have distributed eigenvalues and applying the separation directly to the filters results in significant information loss.”

为了缓解该问题，稳重限制感受野的连接，使得模型在训练阶段能够学习1维的可分滤波器。论文表明通过训练3D空间上各个方向连续的1D滤波器组成的扁平网络，能够在显著降低模型参数的情况下，得到和标准卷积网络相近的性能。


# P6. 1x1/Pointwise Convolutions

1\*1卷积最初在Network-in-network(NiN, https://arxiv.org/abs/1312.4400)中提出，后被广泛用于Inception中（https://arxiv.org/abs/1409.4842）。主要用于通道变换，在不改变输入特征的宽高的情况下，对通道数进行升维或者降维。卷积核大小为1\*1，卷积核参数量为Cout\*Cin\*1\*1，其中Cout个Cin\*1\*1的卷积核和输入特征进行乘法，得到Cout个H\*W的临时结果，并将这些结果拼接，得到Cout\*H\*W的输出结果，在未改变特征宽高的情况下，改变了特征的通道数。

1\*1卷积的优点如下：

- Dimensionality reduction for efficient computations：有效进行维度变换

- Efficient low dimensional embedding, or feature pooling：即便低维嵌入式空间中也可能包含相对较大图像块的大量信息。因而在使用3\*3或5\*5卷积之前先使用1\*1卷积。

- Applying nonlinearity again after convolution：可在1\*1卷积之后增加ReLU等激活函数，让网络学到更复杂的变换。

- Yann LeCun指出全连接层充当1\*1卷积：“In Convolutional Nets, there is no such thing as “fully-connected layers”. There are only convolution layers with 1x1 convolution kernels and a full connection table.”（卷积网络中没有全连接层，只有使用1\*1卷积的层和全连接表）。

![6](/assets/post/2021-07-23-differentconv/6.png)


# P7. 分组卷积（Grouped Convolution）

来自2012年的论文AlexNet（之后在ResNeXt中再次使用，主要动机是通过将特征分组，来减少计算复杂度），主要原因是在显存有限的2个GPU上（每个GPU有1.5GB显存）训练模型。如下图的AlexNet，大多数层上为2个分开的卷积路径。该网络在2个（或多个）GPU上进行模型并行化。

![7\_1](/assets/post/2021-07-23-differentconv/7_1.png)

接下来介绍分组卷积的工作原理。首先，传统的2D卷积工如下：Hin\*Win\*Din的输入通过Dout个h\*w\*Din的卷积核变换到Hout\*Wout\*Dout的输出。

![7\_2](/assets/post/2021-07-23-differentconv/7_2.png)

分组卷积中，卷积核被分成不同的组。每组在通道方向上负责特定深度的传统2D卷积，并将结果拼接，得到最终结果。如下图显示了2组的分组卷积。每组卷积核的深度为Din/2，每组滤波器包含Dout/2个卷积核。第一组（红色）用于处理输入层的前一半数据（[:, :, 0:Din/2]），第二组（橙色）用于处理输入层的后一半数据（[:, :, Din/2:Din]）。因而每组会输出Dout/2个通道。两组结果拼接起来，得到Dout通道的输出。

![7\_3](/assets/post/2021-07-23-differentconv/7_3.png)

分组卷积和深度可分离卷积中的深度卷积的关系：如果分组卷积中的滤波器组数和输入层通道数相同（每个滤波器深度为1），等价于深度卷积；如果每个滤波器组包含Dout/Din个滤波器（输出层深度为Dout），这和深度卷积不同：深度卷积不改变层的深度（深度可分离卷积中，通过1\*1卷积实现通道变换）。

分组卷积的优点：

**① 训练更高效**：每次能使用更多图像训练。模型并行化比数据并行化更有效。

**② 模型更高效**：模型参数随着分组数量的增加而降低。假定分组数为n，则模型包含
$$\left( h*w*\frac{ { {D}_{in}}}{n}*\frac{ { {D}_{out}}}{n} \right)*n$$
个参数，参数数量降低到原来的
$$\frac{1}{n}$$
。

**③ 分组卷积能提供更好的模型**。具体见：<https://blog.yani.io/filter-group-tutorial/>。简单来说和系数滤波器的关系有关。该博客提出了一个原因：The effect of filter groups is to learn with a block-diagonal structured sparsity on the channel dimension… the filters with high correlation are learned in a more structured way in the networks with filter groups. In effect, filter relationships that don’t have to be learned are no longer parameterized. In reducing the number of parameters in the network in this salient way, it is not as easy to over-fit, and hence a regularization-like effect allows the optimizer to learn more accurate, more efficient deep networks.”（滤波器组用于学习在通道维度上对角块结构的洗属性…在有滤波器组的网络中，能以更结构化的方式学习具有更高相关性的滤波器。从效果上看，无需参数化那些不必学习的滤波器关系。通过这种方式显著减少网络参数量时，不易过拟合，且这种类似正则化的方式允许优化器学习更准确、更有效的深度网络。）另一方面，每组滤波器学习了不同性质的滤波器：黑白滤波器和彩色滤波器，如下图：

![7_4](/assets/post/2021-07-23-differentconv/7_4.png)

分组卷积可以极大减少模型参数，同时增大卷积之间的对角相关性，不容易过拟合，相当于正则的效果。


# P8. 混淆分组卷积（Shuffled Grouped Convolution）

由旷视在<https://arxiv.org/abs/1707.01083>中提出（ShuffleNet）。ShuffleNet可用于算力有限的移动设备上(如10–150 MFLOPs)。混淆分组卷积基于分组卷积和宽度可分离卷积（depthwise separable convolution），其包括分组卷积核通道混淆。

下图为有2个堆叠分组卷积的通道混淆示意图。GConv代表分组卷积。a为分组数相同的2个堆叠的分组卷积，由于每组只负责处理前一层传进来的信息因而每个滤波器组只能学习特定的特征，这种方式阻断了不同组之间的信息交换，削弱了模型在训练阶段的表达能力。因而使用通道混淆，来混合不同滤波器组的信息，如b所示，先将GConv1的特征分成不同的子组并组合，然后送入分组卷积GConv2，得到最终的特征。c为等效于b的实现，即先对GConv1输出的特征进行通道混淆，然后再正常输入GConv2。分组卷积中，每个滤波器组学习当前输入的特定特征，阻碍了特征信息在不同组之间的传播，通道混淆则打乱了通道顺序，解决了信息无法传播的问题。

![8](/assets/post/2021-07-23-differentconv/8.png)


# P9. Pointwise grouped convolution逐点分组卷积

ShuffleNet（<https://arxiv.org/abs/1707.01083>）中还提出了逐点分组卷积（pointwise grouped convolution）。MobileNet或ResNeXt中在3\*3卷积中使用分组卷积，而未在1\*1卷积中使用。ShuffleNet认为1\*1卷积计算成本也很高，因而建议1\*1卷积也使用分组卷积。1\*1的分组卷积称作逐点分组卷积（pointwise grouped convolution），能进一步降低计算成本。

文中提出了ShuffleNet单元，如下图。a为使用宽度卷积（depthwise convolution， DWConv）的瓶颈单元。b为使用逐点分组卷积（pointwise group convolution）和通道混淆的ShuffleNet单元。c为当stride=2时的ShuffleNet单元，其中将最后的操作由add改为了concat。

![9](/assets/post/2021-07-23-differentconv/9.png)


# P10. Spatial and Cross-Channel Convolutions空间和跨通道卷积
Inception网络中广泛使用。主要将跨通道的相关性和空间相关性拆分成一系列独立的操作。空间相关性指在宽、高方向上使用卷积，如下图的3\*3卷积。如下图，使用3个独立的“1\*1卷积+3\*3卷积”分别处理通道相关性。

![10](/assets/post/2021-07-23-differentconv/10.png)


# P11. 可分卷积（Separable Convolutions）

可分卷积在MobileNet（<https://arxiv.org/abs/1704.04861>）等网络中使用，其包括空间可分卷积（Spatially Separable Convolutions）和深度可分卷积（Depthwise Separable Convolutions）

## P11.1. 空间可分卷积 Spatially Separable Convolutions
空间可分卷积在图像（或特征）的宽高（WH）维度上进行卷积。从概念上来说，空间可分离卷积将一个卷积分解为两个单独的运算。如3\*3的Sobel核可被分解成1个3\*1的核和一个1\*3的核，如下图。传统卷积中，3\*3的核直接和图像卷积。空间可分离卷积中，先将3\*1的核和图像卷积，再将1\*3的核和图像卷积。在执行相同操作的情况下，将参数量从9降低到6。

$$\left[ \begin{matrix}
   \text{-}1 & 0 & 1  \\
   \text{-}2 & 0 & 2  \\
   \text{-}1 & 0 & 1  \\
\end{matrix} \right]\text{=}\left[ \begin{matrix}
   1  \\
   2  \\
   1  \\
\end{matrix} \right]\times \left[ \begin{matrix}
   \text{-}1 & 0 & 1  \\
\end{matrix} \right]$$ 

另一方面，空间可分离卷积也可以减少矩阵乘法次数。对于N\*N的单通道图像和m\*m的卷积核，在stride=1，padding=0时，传统卷积需要(N-2)\*(N-2)\*m\*m次乘法，空间可分离卷积需要N\*(N-2)\*m+(N-2)\*(N-2)\*m=(2N-2) x (N-2) x m次乘法。空间可分离卷积和标准卷积的计成本之比为：

$$\frac{2}{m}+\frac{2}{m(N-2)}$$

当图像宽高N远大于卷积核大小m（N >> m）时，该比例近似为2/m。
虽然空间可分离卷积节省计算代价，但是在深度学习中很少使用。一个主要原因是不是所有卷积核都能被分解成2个更小的卷积核。若将传统卷积替换为空间可分离卷积，在训练中会限制搜索到的所有可能的卷积核。导致训练得到次优模型。

## P11.2. 深度可分离卷积 Depthwise Separable Convolutions

深度可分离卷积在深度学习中更常使用，如MobileNet and Xception。深度可分离卷积包括2步：深度卷积（depthwise convolutions）和1\*1卷积。

深度可分离卷积可见：<https://www.cnblogs.com/darkknightzh/p/9410540.html>

在对
$$\left[ { {C}_{in}},{ {H}_{in}},{ {W}_{in}} \right]$$
的特征进行传统卷积时（stride=1，padding=0），卷积核共
$${ {C}_{out}}*{ {C}_{in}}*{ {H}_{k}}*{ {W}_{k}}$$
个参数，为
$${ {C}_{out}}$$
个
$$\left[ { {C}_{in}},{ {H}_{k}},{ {W}_{k}} \right]$$
的卷积核和输入特征做卷积，得到
$${ {C}_{out}}$$
个
$$\left[ 1,{ {H}_{in}}-2,{ {W}_{in}}-2 \right]$$
的临时结果；并将
$${ {C}_{out}}$$
个临时结果拼接成
$$\left[ { {C}_{out}},{ {H}_{in}}-2,{ {W}_{in}}-2 \right]$$
的输出特征。

在对
$$\left[ { {C}_{in}},{ {H}_{in}},{ {W}_{in}} \right]$$
的特征进行深度可分离卷积时（stride=1，padding=0），首先使用
$${ {C}_{in}}$$
个
$$\left[ 1,{ {H}_{k}},{ {W}_{k}} \right]$$
的卷积核和输入的
$${ {C}_{in}}$$
个
$$\left[ 1,{ {H}_{in}},{ {W}_{in}} \right]$$
的特征分别进行卷积，得到
$${ {C}_{in}}$$
个临时结果，并将这
$${ {C}_{in}}$$
个临时结果拼接，得到
$$\left[ { {C}_{in}},{ {H}_{in}}-2,{ {W}_{in}}-2 \right]$$
的临时结果2。此时通道数（特征的深度）不变。之后使用
$${ {C}_{out}}$$
个
$$\left[ { {C}_{in}},1,1 \right]$$
的卷积核，和临时结果2分别进行卷积，得到
$${ {C}_{out}}$$
个
$$\left[ 1,{ {H}_{in}}-2,{ {W}_{in}}-2 \right]$$
的结果，并在通道维度拼接，得到
$$\left[ { {C}_{out}},{ {H}_{in}}-2,{ {W}_{in}}-2 \right]$$
的输出，该结果维度和传统卷积维度一样。

深度可分离卷积的优点：高效。与2D卷积相比，深度可分离卷积需要更少的操作。

对于
$$\left[ { {C}_{in}},{ {H}_{in}},{ {W}_{in}} \right]$$
的特征，stride=1，padding=0时，传统卷积的乘法次数：

$${ {C}_{out}}\times { {C}_{in}}\times { {H}_{k}}\times { {W}_{k}}\times \left( { {H}_{in}}-{ {H}_{k}}+1 \right)\times \left( { {W}_{in}}-{ {W}_{k}}+1 \right)$$

深度卷积乘法次数：
$${ {C}_{in}}\times { {H}_{k}}\times { {W}_{k}}\times 1\times \left( { {H}_{in}}-{ {H}_{k}}+1 \right)\times \left( { {W}_{in}}-{ {W}_{k}}+1 \right)$$
，1*1卷积乘法次数：
$${ {C}_{in}}\times 1\times 1\times { {C}_{out}}\times \left( { {H}_{in}}-{ {H}_{k}}+1 \right)\times \left( { {W}_{in}}-{ {W}_{k}}+1 \right)$$
，因而深度可分离卷积乘法次数：

$${ {C}_{in}}\times { {H}_{k}}\times { {W}_{k}}\times 1\times \left( { {H}_{in}}-{ {H}_{k}}+1 \right)\times \left( { {W}_{in}}-{ {W}_{k}}+1 \right)+{ {C}_{in}}\times 1\times 1\times { {C}_{out}}\times \left( { {H}_{in}}-{ {H}_{k}}+1 \right)\times \left( { {W}_{in}}-{ {W}_{k}}+1 \right)$$

深度可分离卷积和原始卷积乘法次数之比为：

$$\frac{1}{ { {C}_{out}}}+\frac{1}{ { {H}_{k}}*{ {W}_{k}}}$$

当输出通道数Cout远大于卷积核宽高时，上式近似为
$$1/\left( { {H}_{k}}\times { {W}_{k}} \right)$$
，意味着当时用3\*3卷积核时，2D卷积乘法次数为深度可分离卷积的9倍。

深度可分离卷积的缺点：深度可分离卷积降低了模型的参数，在模型比较小时，相比2D卷积的性能可能会降低的很明显，导致模型模型次优。但是，如果使用恰当的话，深度可分离卷积能在不显著降低模型性能的前提下，提高模型效率。

**Inception模块和可分卷积的区别**：a 可分卷积先在通道上使用空间卷积，然后使用1\*1conv。Inception模块先使用1\*1conv。b 可分卷积通常不使用非线性层。


# P12. 可变性卷积（Deformable Convolution）

可变形卷积是在基础卷积核的上添加一些位移量，根据数据的学习情况，自动调整偏移，卷积核可以在任意方向进行伸缩，改变感受野的范围，该位移量是靠额外的一个卷积层进行自动学习的，如下图，（a）是普通的卷积，卷积核大小为3\*3，采样点排列非常规则，是一个正方形。（b）是可变形的卷积，给每个采样点加一个offset（这个offset通过额外的卷积层学习得到），排列变得不规则。（c）和（d）是可变形卷积的两种特例。对于（c）加上offset，达到尺度变换的效果；对于（d）加上offset，达到旋转变换的效果。 

![12\_1](/assets/post/2021-07-23-differentconv/12_1.png)

某一点P0经过标准卷积后结果为 

$$y\left( { {P}_{0}} \right)=\sum\limits_{ { {P}_{n}}\in R}{w\left( { {P}_{n}} \right)\centerdot x\left( { {P}_{0}}+{ {P}_{n}} \right)}$$

而经过可变形卷积后的结果为： 

$$y\left( { {P}_{0}} \right)=\sum\limits_{ { {P}_{n}}\in R}{w\left( { {P}_{n}} \right)\centerdot x\left( { {P}_{m}} \right)}$$

这里
$${ {P}_{m}}={ {P}_{0}}+{ {P}_{n}}+\Delta { {P}_{n}}$$
，
$$\Delta { {P}_{n}}$$
为卷积核的偏移量offset。可变形卷积更能适应目标的各种形变。

就特征提取的形状而言，卷积非常严格。也就是说，kernel形状仅为正方形/矩形（或其他一些需要手动确定的形状），因此它们只能在这种模式下使用。如果卷积的形状本身是可学习的呢？这是引入可变形卷积背后的核心思想。

![12\_2](/assets/post/2021-07-23-differentconv/12_2.jpg)

实际上，可变形卷积的实现非常简单。每个kernel都用两个不同的矩阵表示。第一分支学习从原点预测“偏移”。此偏移量表示要处理原点周围的哪些输入。由于每个偏移量都是独立预测的，它们之间无需形成任何刚性形状，因此具有可变形的特性。第二个分支只是卷积分支，其输入是这些偏移量处的值。

![12\_3](/assets/post/2021-07-23-differentconv/12_3.jpg)
