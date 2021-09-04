---
layout: post
title:  "Softmax导数的计算"
date:   2021-07-02 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/gradientSoftmax>


假设softmax层输入特征为
$$\mathbf{z}=[{ {z}_{1}},\cdots ,{ {z}_{i}},\cdots ,{ {z}_{n}}]$$
，softmax层输出特征为
$$\mathbf{a}=[{ {a}_{1}},\cdots ,{ {a}_{i}},\cdots ,{ {a}_{n}}]$$
，实际标签
$$\mathbf{y}=[{ {y}_{1}},\cdots ,{ {y}_{i}},\cdots ,{ {y}_{n}}]$$
，其中
$${ {a}_{i}}=\frac{ { {e}^{ { {z}_{i}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}$$
，n为特征数量。如下图所示。

![1](/assets/post/2021-07-03-gradientSoftmax/1softmax.png)
_图1_
 
使用交叉熵多标签分类时得到的损失为L，n为类别数，则

$$L=-\sum\limits_{k=1}^{n}{ { {y}_{k}}\ln { {a}_{k}}}$$

假定
$${ {y}_{k}}$$
中只有第j类的标签
$${ {y}_{j}}=1$$
，其他标签均为0。则
$$L=-\ln { {a}_{j}}$$

损失L关于输入z的偏导数计算如下：

$$\frac{\partial L}{\partial { {z}_{i}}}\text{=}\frac{\partial L}{\partial { {a}_{j}}}\frac{\partial { {a}_{j}}}{\partial { {z}_{i}}}\text{=}\frac{\partial \left( -\ln { {a}_{j}} \right)}{\partial { {a}_{j}}}\bullet \frac{\partial { {a}_{j}}}{\partial { {z}_{i}}}\text{=}-\frac{1}{ { {a}_{j}}}\bullet \frac{\partial { {a}_{j}}}{\partial { {z}_{i}}}$$

① 当j=i时，

$$\frac{\partial L}{\partial { {z}_{i}}}\text{=}-\frac{1}{ { {a}_{i}}}\bullet \frac{\partial { {a}_{i}}}{\partial { {z}_{i}}}$$

先计算

$$\begin{align}
  & \frac{\partial { {a}_{i}}}{\partial { {z}_{i}}}\text{=}\frac{\partial \frac{ { {e}^{ { {z}_{i}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}}{\partial { {z}_{i}}}=\frac{\frac{\partial { {e}^{ { {z}_{i}}}}}{\partial { {z}_{i}}}\centerdot \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}-{ {e}^{ { {z}_{i}}}}\centerdot \frac{\partial \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}{\partial { {z}_{i}}}}{ { {\left( \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}} \right)}^{2}}} \\ 
 & \xrightarrow{\frac{\partial \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}{\partial { {z}_{i}}}\text{只有}k=i\text{时为}{ {e}^{ { {z}_{i}}}}\text{，其他均为}0}\frac{ { {e}^{ { {z}_{i}}}}\centerdot \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}-{ {e}^{ { {z}_{i}}}}\centerdot { {e}^{ { {z}_{i}}}}}{ { {\left( \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}} \right)}^{2}}} \\ 
 & \text{=}\frac{ { {e}^{ { {z}_{i}}}}\centerdot \left( \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}-{ {e}^{ { {z}_{i}}}} \right)}{ { {\left( \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}} \right)}^{2}}}=\frac{ { {e}^{ { {z}_{i}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}\centerdot \frac{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}-{ {e}^{ { {z}_{i}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}} \\ 
 & =\frac{ { {e}^{ { {z}_{i}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}\centerdot \left( 1-\frac{ { {e}^{ { {z}_{i}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}} \right)={ {a}_{i}}\centerdot \left( 1-{ {a}_{i}} \right) \\ 
\end{align}$$

因而

$$\frac{\partial L}{\partial { {z}_{i}}}\text{=}-\frac{1}{ { {a}_{i}}}\bullet \frac{\partial { {a}_{i}}}{\partial { {z}_{i}}}\text{=}-\frac{1}{ { {a}_{i}}}\centerdot { {a}_{i}}\centerdot \left( 1-{ {a}_{i}} \right)\text{=}{ {a}_{i}}\text{-}1$$

② 当j≠i时：

$$\frac{\partial L}{\partial { {z}_{i}}}\text{=}-\frac{1}{ { {a}_{j}}}\bullet \frac{\partial { {a}_{j}}}{\partial { {z}_{i}}}$$

先计算

$$\begin{align}
  & \frac{\partial { {a}_{j}}}{\partial { {z}_{i}}}\text{=}\frac{\partial \frac{ { {e}^{ { {z}_{j}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}}{\partial { {z}_{i}}}\xrightarrow{ { {e}^{ { {z}_{j}}}}\text{与}{ {z}_{i}}\text{无关}}-\frac{ { {e}^{ { {z}_{j}}}}\centerdot \frac{\partial \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}{\partial { {z}_{i}}}}{ { {\left( \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}} \right)}^{2}}} \\ 
 & \xrightarrow{\frac{\partial \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}{\partial { {z}_{i}}} \text{只有}k=i\text{时为}{ {e}^{ { {z}_{i}}}}\text{，其他均为}0}-\frac{ { {e}^{ { {z}_{j}}}}\centerdot { {e}^{ { {z}_{i}}}}}{ { {\left( \sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}} \right)}^{2}}} \\ 
 & \text{=}-\frac{ { {e}^{ { {z}_{j}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}}\centerdot \frac{ { {e}^{ { {z}_{i}}}}}{\sum\limits_{k=1}^{n}{ { {e}^{ { {z}_{k}}}}}} \\ 
 & =-{ {a}_{j}}\centerdot { {a}_{i}} \\ 
\end{align}$$

因而

$$\frac{\partial L}{\partial { {z}_{i}}}\text{=}-\frac{1}{ { {a}_{j}}}\bullet \left( -{ {a}_{j}}\centerdot { {a}_{i}} \right)\text{=}{ {a}_{i}}$$

综上：

$$\frac{\partial L}{\partial { {z}_{i}}}\text{=}\left\{ \begin{matrix}
   { {a}_{i}}\text{-}1 & j=i  \\
   { {a}_{i}} & j\ne i  \\
\end{matrix} \right.$$

其中j为label为1的样本索引。

由于
$$\mathbf{a}=[{ {a}_{1}},\cdots ,{ {a}_{i}},\cdots ,{ {a}_{n}}]$$
，
$$\mathbf{y}=[{ {y}_{1}},\cdots ,{ {y}_{i}},\cdots ,{ {y}_{n}}]\text{=}[0,\cdots ,1,\cdots ,0]$$
，转换成向量形式

$$\frac{\partial L}{\partial \mathbf{z}}\text{=}\mathbf{a}-\mathbf{y}$$
