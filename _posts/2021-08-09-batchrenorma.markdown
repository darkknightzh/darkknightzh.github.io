---
layout: post
title:  "Batch Renormalization（BRN）"
date:   2021-08-09 16:00:00 +0800
tags: [deep learning, algorithm, normalization]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/batchrenorma>


论文：

<https://arxiv.org/abs/1702.03275>

<br>

在训练和测试阶段使用BN时，由于归一化方式不同，因而网络的行为也不同。为了解决这个问题，作者提出了Batch Renormalization（BRN），即先对输入x进行仿射变换，然后再通过BN，得到输出。

假设网络中某节点为x，其通过最后几次batch得到的均值和标准差分别为
$$\mu $$
和
$$\sigma $$
，则

$$\frac{ { {x}_{i}}-\mu }{\sigma }=\frac{ { {x}_{i}}-{ {\mu }_{B}}}{ { {\sigma }_{B}}}\centerdot r+d \quad \quad \text{其中}r=\frac{ { {\sigma }_{B}}}{\sigma },\text{  }d=\frac{ { {\mu }_{B}}-\mu }{\sigma }$$

如果
$$\sigma \text{=}E\left[ { {\sigma }_{B}} \right]$$
且
$$\mu \text{=}E\left[ { {\mu }_{B}} \right]$$
，则
$$E\left[ r \right]=1$$
,
$$E\left[ d \right]=0$$
。当r=1且d=0时，上式即为BN。

我们保持r和d，但是为了计算梯度，将它们视为常数。我们加强一个网络，该网络包含BN层，和一个应用仿射变换进行归一化的激活层。该仿射变换的r和d是固定的（虽然它们通过当前batch计算）。结合该仿射变换的BN称作Batch Renormalization（BRN），保证训练和测试的行为一致。

实际上可以先使用BN，不使用矫正训练；一段时间后加大矫正的范围。通过在r和d上增加边界限制来实现，初始时限制它们分别为1和0，而后逐步放松限制。

BRN训练阶段也使用
$$\mu $$
和
$$\sigma $$
，来进行矫正。使用指数延迟滑动平均的方式更新μ和σ，并优化模型剩下的参数。BRN确保训练中使用的参数在测试中能直接使用。BRN算法如下：

**Input**: Values of 
$$x$$
over a mini-batch: 
$$B=\{ { {x}_{1\cdots m}}\}$$
;

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters 
$$\gamma $$
, 
$$\beta $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;current moving mean $$\mu $$ and standard deviation $$\sigma $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;moving average update rate $$\alpha $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;maximum allowed correction $${ {r}_{\max }}$$, $${ {d}_{\max }}$$

**Output**: $$\left\{ { {y}_{i}}=BatchRenorm({ {x}_{i}}) \right\}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;update 
$$\mu $$
, 
$$\sigma $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$${ {\mu }_{B}}\leftarrow \frac{1}{m}\sum\limits_{i=1}^{m}{ { {x}_{i}}}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$${ {\sigma }_{B}}\leftarrow \sqrt{\varepsilon +\frac{1}{m}\sum\limits_{i=1}^{m}{ { {\left( { {x}_{i}}-{ {\mu }_{B}} \right)}^{2}}}}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$r\leftarrow stop\_gradient\left( cli{ {p}_{\left[ 1/{ {r}_{\max }},{ {r}_{\max }} \right]}}\left( \frac{ { {\sigma }_{B}}}{\sigma } \right) \right)$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$d\leftarrow stop\_gradient\left( cli{ {p}_{\left[ -{ {d}_{\max }},{ {d}_{\max }} \right]}}\left( \frac{ { {\mu }_{B}}-\mu }{\sigma } \right) \right)$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$${ {\hat{x}}_{i}}\leftarrow \frac{ { {x}_{i}}-{ {\mu }_{B}}}{ { {\sigma }_{B}}}\centerdot r+d$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$${ {y}_{i}}\leftarrow \gamma { {\hat{x}}_{i}}+\beta $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$\mu :=\mu +\alpha \left( { {\mu }_{B}}-\mu  \right)$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// 使用相当大的$\alpha $更新$$\mu $$和$$\sigma $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$\sigma :=\sigma +\alpha \left( { {\sigma }_{B}}-\sigma  \right)$$

**Inference**：
$$y\leftarrow \gamma \centerdot \frac{x-\mu }{\sigma }+\beta $$
