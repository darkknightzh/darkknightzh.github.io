---
layout: post
title:  "Batch Normalization（BN）"
date:   2021-08-09 16:00:00 +0800
tags: [deep learning, algorithm, normalization]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/batchnorm>

<br>

**(1) Internal Covariate Shift**

在训练过程中，每次参数更新后，上层网络的输出通过这层网络后，数据的分布会发生变化，导致训练困难，这就是Internal Covariate Shift。 Internal指的是深层网络内部的中间层。与其对应的是covariate shift发生在输入层，指训练数据和测试数据分布的差异性，导致网络泛化性能的降低。可以通过归一化或者白化来解决Covariate Shift。而对于Internal Covariate Shift，则通过BN来解决。BN让深度网络的每一层输入保持相同分布，来解决Internal Covariate Shift。

**(2) BatchNorm的本质思想**

由于在训练过程中，随着网络的加深，数据的分布会逐渐往激活函数的上下限两端靠近（饱和区），导致反向传播时低层梯度消失，影响了网络收敛速度。BN则把每个神经元输入的分布强行拉回均值0方差为1的标准正态分布，确保激活函数的输入落到线性区，从而增大反传时的梯度，避免梯度消失，并且加快收敛速度。

如果直接使用多层线性函数没有意义，因为多层线性函数等价于一层线性函数。因而为了保证非线性的获得，BN对变换后的分布又进行了缩放（
$$*\gamma $$
）和平移（
$$+\beta $$
），
$$\gamma $$
和
$$\beta $$
这两个参数通过学习得到，让每层的输入不是标准正态分布，保证输入能有一定的非线性。

**(3) 训练阶段**

公式如下：

$${ {y}^{(k)}}={ {\gamma }^{(k)}}{ {\hat{x}}^{(k)}}+{ {\beta }^{(k)}}$$

具体的算法流程如下:

**Input**: Values of 
$$x$$
 over a mini-batch: 
$$B=\{ { {x}_{1\cdots m}}\}$$
;

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters to be learned:
$$\gamma ,\beta $$

**Output**: 
$$\left\{ { {y}_{i}}=B{ {N}_{\gamma ,\beta }}({ {x}_{i}}) \right\}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$${ {\mu }_{B}}\leftarrow \frac{1}{m}\sum\limits_{i=1}^{m}{ { {x}_{i}}}$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// mini-batch mean

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$\sigma _{B}^{2}\leftarrow \frac{1}{m}\sum\limits_{i=1}^{m}{ { {\left( { {x}_{i}}-{ {\mu }_{B}} \right)}^{2}}}$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// mini-batch variance

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$${ {\hat{x}}_{i}}\leftarrow \frac{ { {x}_{i}}-{ {\mu }_{B}}}{\sqrt{\sigma _{B}^{2}+\varepsilon }}$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // normalize

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$${ {y}_{i}}\leftarrow \gamma { {\hat{x}}_{i}}+\beta \equiv B{ {N}_{\gamma ,\beta }}({ {x}_{i}})$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// scale and shift

反向传播的导数计算公式：

$$\frac{\partial l}{\partial { { {\hat{x}}}_{i}}}=\sum\limits_{j=1}^{m}{\frac{\partial l}{\partial { {y}_{j}}}\centerdot \frac{\partial { {y}_{j}}}{\partial { { {\hat{x}}}_{i}}}}\xrightarrow{j=i\text{时}{ {y}_{j}}\text{和}{ { {\hat{x}}}_{i}}\text{有关}}\frac{\partial l}{\partial { {y}_{i}}}\centerdot \frac{\partial { {y}_{i}}}{\partial { { {\hat{x}}}_{i}}}=\frac{\partial l}{\partial { {y}_{i}}}\centerdot \gamma $$
&nbsp;

$$\frac{\partial l}{\partial \sigma _{B}^{2}}=\sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { { {\hat{x}}}_{i}}}\centerdot \frac{\partial { { {\hat{x}}}_{i}}}{\partial \sigma _{B}^{2}}}=\sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { { {\hat{x}}}_{i}}}\centerdot \left( { {x}_{i}}-{ {\mu }_{B}} \right)\centerdot \frac{-1}{2}\centerdot { {\left( \sigma _{B}^{2}+\varepsilon  \right)}^{-\frac{3}{2}}}}$$
&nbsp;

$$\begin{align}
  & \frac{\partial l}{\partial { {\mu }_{B}}}\xrightarrow{ { {\mu }_{B}} \text{分别和}{ { {\hat{x}}}_{i}}\text{及}\sigma _{B}^{2}\text{有关}}\left( \sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { { {\hat{x}}}_{i}}}\centerdot \frac{\partial { { {\hat{x}}}_{i}}}{\partial { {\mu }_{B}}}} \right)\text{+}\frac{\partial l}{\partial \sigma _{B}^{2}}\centerdot \frac{\partial \sigma _{B}^{2}}{\partial { {\mu }_{B}}} \\ 
 & \quad\quad=\left( \sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { { {\hat{x}}}_{i}}}\centerdot \frac{\text{-}1}{\sqrt{\sigma _{B}^{2}+\varepsilon }}} \right)\text{+}\frac{\partial l}{\partial \sigma _{B}^{2}}\centerdot \frac{\sum\limits_{i=1}^{m}{(-2)\centerdot \left( { {x}_{i}}-{ {\mu }_{B}} \right)}}{m} \\ 
\end{align}$$
&nbsp;

$$\begin{align}
  & \frac{\partial l}{\partial { {x}_{i}}}\xrightarrow{ { {x}_{i}}\text{分别和}{ { {\hat{x}}}_{i}}\text{、}\sigma _{B}^{2}\text{、}{ {\mu }_{B}}\text{有关}}\left( \sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { { {\hat{x}}}_{i}}}\centerdot \frac{\partial { { {\hat{x}}}_{i}}}{\partial { {x}_{i}}}} \right)\text{+}\frac{\partial l}{\partial \sigma _{B}^{2}}\centerdot \frac{\partial \sigma _{B}^{2}}{\partial { {x}_{i}}}\text{+}\frac{\partial l}{\partial { {\mu }_{B}}}\centerdot \frac{\partial { {\mu }_{B}}}{\partial { {x}_{i}}} \\ 
 & \quad\quad=\frac{\partial l}{\partial { { {\hat{x}}}_{i}}}\centerdot \frac{1}{\sqrt{\sigma _{B}^{2}+\varepsilon }}\text{+}\frac{\partial l}{\partial \sigma _{B}^{2}}\centerdot \frac{2\left( { {x}_{i}}-{ {\mu }_{B}} \right)}{m}+\frac{\partial l}{\partial { {\mu }_{B}}}\centerdot \frac{1}{m} \\ 
\end{align}$$
&nbsp;

$$\frac{\partial l}{\partial \gamma }=\sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { {y}_{i}}}\centerdot \frac{\partial { {y}_{i}}}{\partial \gamma }}=\sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { {y}_{i}}}\centerdot { { {\hat{x}}}_{i}}}$$
&nbsp;

$$\frac{\partial l}{\partial \beta }=\sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { {y}_{i}}}\centerdot \frac{\partial { {y}_{i}}}{\partial \beta }}=\sum\limits_{i=1}^{m}{\frac{\partial l}{\partial { {y}_{i}}}}$$
&nbsp;

**(4) 测试阶段**

在测试阶段，bn的上述参数使用训练阶段统计到的值，测试阶段不更新各参数。

**(5) BatchNorm的优点**

① 极大提升了训练速度，加快收敛速度。

② 提高分类性能，类似于Dropout的正则化方法，防止过拟合。

③ 简化调参过程，降低初始化要求，可以使用更大的学习率。

④ 可认为batchnorm通过去相关的过程，降低数据之间的绝对差异，更多的考虑相对差异性，因此在分类任务上具有更好的效果。

**(6) BatchNorm的缺点**

① Batch Size过小时BN效果明显下降。

由于BN是统计当前batch中样本的均值方差等，batch size比较小则任务效果有明显的下降。下图给出了在ImageNet上使用ResNet分类时，模型性能随batch size的变化情况。可见batch size小于8之后，模型性能显著降低。如果batch size为1，则方差为0，此时BN无法工作。另外，如果batch size太小，会导致噪声太大，影响模型的训练。

![1](/assets/post/2021-08-09-batchnorm/1bn.png)
_图1_

② RNN网络使用BN效果不佳

RNN网络中，每个时间步有不同的意义，意味着需要对每个时间步使用一个bn层。由于要维护每个时间步上的统计特征，导致模型更复杂且更占空间。
