---
layout: post
title:  "Layer Normalization（LN）"
date:   2021-08-09 16:00:00 +0800
tags: [deep learning, algorithm, normalization]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/layernorm>

论文：

<https://arxiv.org/abs/1607.06450>

<br>

假设第l层第i个输入为
$$a_{i}^{l}$$
，忽略偏置时，令
$$\overline{a}_{i}^{l}$$
为l层第i个隐含单元的归一化输入，
$${ {g}^{i}}$$
为增益，即

$$\overline{a}_{i}^{l}=\frac{g_{i}^{l}}{\sigma _{i}^{l}}\left( a_{i}^{l}-\mu _{i}^{l} \right)$$

当时用BN归一化时，

$$\mu _{i}^{l}=\underset{x\sim P\left( x \right)}{ \text{E} }\,\left[ a_{i}^{l} \right]$$

$$\sigma _{i}^{l}=\sqrt{\underset{x\sim P\left( x \right)}{ \text{E} }\,\left[ { {\left( a_{i}^{l}-\mu _{i}^{l} \right)}^{2}} \right]}$$

当时用LN归一化时，

$$\mu _{i}^{l}=\frac{1}{H}\sum\limits_{i=1}^{H}{a_{i}^{l}}$$

$$\sigma _{i}^{l}=\sqrt{\frac{1}{H}\sum\limits_{i=1}^{H}{ { {\left( a_{i}^{l}-\mu _{i}^{l} \right)}^{2}}}}$$

其中H为当前层隐含单元的个数。

LN和BN的区别是，LN中，每一层的所有隐含单元共享相同的归一化μ和σ，并且不同的数据使用不同的归一化项（确保与batch size无关）。因而LN中的batch size可以为1。

个人理解：batch size中的每个训练数据，在当前层会得到相应的输出，假设特征为NCHW。BN使用NHW的特征，得到C个均值和C个方差，以及C个缩放系数和平移系数； LN则是使用CHW的特征，计算N个样本当前特征各自的均值和方差（各N个），以及C\*H\*W个缩放系数和C\*H\*W个平移系数，每个样本当前层减去对应的均值和方差，因而LN和batch size无关。

如下代码，

```python
import torch
import torch.nn as nn

input = torch.randn(20, 5, 10, 10)
m = nn.LayerNorm(input.size()[1:])
print('LayerNorm weight.shape:', m.weight.shape)
print('LayerNorm bias.shape:', m.bias.shape)
```

输出如下：

```terminal
LayerNorm weight.shape: torch.Size([5, 10, 10])
LayerNorm bias.shape: torch.Size([5, 10, 10])
```

由于LN每次统计当前层的所有输入的均值和方差，因而LN不需要额外存储μ和σ，而BN需要额外存储μ和σ。

另外，NLP中不同样本长度不同。使用BN时需要存储序列中每个时间步的统计信息，无法处理测试序列比所有训练序列都长时的情况。由于LN的归一化项只依赖于当前时间步的输入之和，因而LN不会遇到这种问题。另外，LN所有时间步使用相同的增益和偏置参数。

在标准RNN中，给定当前层输入
$${ {\mathbf{x}}^{t}}$$
和前一个时间步的隐含状态
$${ {\mathbf{h}}^{t-1}}$$
，令
$${ {\mathbf{W}}_{hh}}$$
为隐含层到隐含层的权重，
$${ {\mathbf{W}}_{xh}}$$
为输入层到隐含层的权重，则当前时间步的输出为：

$${ {\mathbf{a}}^{t}}={ {\mathbf{W}}_{hh}}{ {\mathbf{h}}^{t-1}}+{ {\mathbf{W}}_{xh}}{ {\mathbf{x}}^{t}}$$

使用LN时，输出如下：

$${ {\mathbf{h}}^{t}}=f\left[ \frac{\mathbf{g}}{ { {\sigma }^{t}}}\odot \left( { {\mathbf{a}}^{t}}-{ {\mu }^{t}} \right)+\mathbf{b} \right]$$

$${ {\mu }^{t}}=\frac{1}{H}\sum\limits_{i=1}^{H}{a_{i}^{t}}$$

$${ {\sigma }^{t}}=\sqrt{\frac{1}{H}\sum\limits_{i=1}^{H}{ { {\left( a_{i}^{t}-{ {\mu }^{t}} \right)}^{2}}}}$$

其中
$$\odot $$
为两个向量的逐元素乘法。b和g分别为偏置和增益项，和
$${ {\mathbf{h}}^{t}}$$
维度相同。而
$${ {\mu }^{t}}$$
和
$${ {\sigma }^{t}}$$
均为标量。

<https://zhuanlan.zhihu.com/p/54530247>中指出：

① 全连接层（FC）上，batch size比较大时，BN优于LN；batch size比较小时，LN优于BN。

② RNN使用LN优于BN。

③ CNN使用LN无法收敛，所以CNN更倾向于使用BN。
