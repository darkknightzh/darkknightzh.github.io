---
layout: post
title:  "池化反向传播公式的推导"
date:   2021-07-09 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/backpropPool>


参考网址：

<https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/>

<http://coollwd.top/articles/2019/12/31/1577771360985.html>


损失函数C对当前点
$${ {x}_{ij}}$$
的导数如下

$$\frac{\partial C}{\partial { {x}_{ij}}}=\frac{\partial C}{\partial { {y}_{uv}}}\frac{\partial { {y}_{uv}}}{\partial { {x}_{ij}}} \tag{1}$$

上式等号右侧并无求和，因为pooling是下采样，输入多个点对应输出一个点，即输入的某个点只和输出的一个点有关（在步长stride=池化块宽高kernel size时）。其中
$$\frac{\partial C}{\partial { {y}_{uv}}}$$
为损失C对当前特征池化后的误差。令其为
$${ {\delta }_{uv}}$$
，即

$${ {\delta }_{uv}}=\frac{\partial C}{\partial { {y}_{uv}}} \tag{2}$$

则公式(1)等效于：

$$\frac{\partial C}{\partial { {x}_{ij}}}={ {\delta }_{uv}}\frac{\partial { {y}_{uv}}}{\partial { {x}_{ij}}} \tag{3}$$  

## P1. max pooling

$${ {y}_{uv}}=\underset{0\le s<{ {k}_{1}},\text{ }0\le t<{ {k}_{2}}}{\mathop \max }\,({ {x}_{i+s,j+t}})={ {x}_{p,q}} \tag{4}$$

因而公式(3)第二项:

$$\frac{\partial { {y}_{uv}}}{\partial { {x}_{ij}}}\text{=}\frac{\partial { {x}_{p,q}}}{\partial { {x}_{ij}}}\text{=}\left\{ \begin{matrix}
   1 & p=i,q=j \text{。即当前点为最大值点}  \\
   0 & \text{其他}  \\
\end{matrix} \right. \tag{5}$$

此处因为只有输入池化块最大值的位置和输出有关。

公式(5)带入公式(3):

$$\frac{\partial C}{\partial { {x}_{ij}}}\text{=}{ {\delta }_{uv}}\frac{\partial { {y}_{uv}}}{\partial { {x}_{ij}}}=\left\{ \begin{matrix}
   { {\delta }_{uv}} & \text{当前点为最大值点}  \\
   0 &  \text{其他} \\
\end{matrix} \right. \tag{6}$$

如下图所示，反传时只有最大值的位置处的梯度为max pooling后的误差，其他位置误差均为0。

![1](/assets/post/2021-07-09-backpropPool/1.png)
_图1_

## P2. avg pooling

$${ {y}_{uv}}=\frac{1}{ { {k}_{1}}\times { {k}_{2}}}\sum\limits_{s=0}^{ { {k}_{1}}}{\sum\limits_{t=0}^{ { {k}_{2}}}{({ {x}_{u+s,v+t}})}} \tag{7}$$

因而公式(3)第二项:

$$\frac{\partial { {y}_{uv}}}{\partial { {x}_{ij}}}\text{=}\frac{\partial }{\partial { {x}_{ij}}}\left( \frac{1}{ { {k}_{1}}\times { {k}_{2}}}\sum\limits_{s=0}^{ { {k}_{1}}}{\sum\limits_{t=0}^{ { {k}_{2}}}{({ {x}_{u+s,v+t}})}} \right)\xrightarrow{u+s=i,v+t=t\text{时，}{ {x}_{u+s,v+t}}\text{与}{ {x}_{ij}}\text{有关}}\frac{1}{ { {k}_{1}}\times { {k}_{2}}} \tag{8}$$

此处因为输入池化块的每个位置都和输出有关。

公式(8)带入公式(3):

$$\frac{\partial C}{\partial { {x}_{ij}}}\text{=}{ {\delta }_{uv}}\frac{\partial { {y}_{uv}}}{\partial { {x}_{ij}}}=\frac{ { {\delta }_{uv}}}{ { {k}_{1}}\times { {k}_{2}}} \tag{9}$$

如下图所示，反传时池化块每个位置的梯度均为avg pooling后的误差乘以
$$\frac{1}{ { {k}_{1}}\times { {k}_{2}}}$$
的系数，其中k1和k2为池化块的宽高。
 
![2](/assets/post/2021-07-09-backpropPool/2.png)
_图2_
