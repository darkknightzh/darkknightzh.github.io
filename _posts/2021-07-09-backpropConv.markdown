---
layout: post
title:  "卷积反向传播公式的推导"
date:   2021-07-09 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/backpropConv>


参考网址：

<https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/>

说明：本文未考虑padding和stride


## P1. 变量定义

假定输入特征为
$$M\times N$$
，卷积核大小为
$${ {k}_{1}}\times { {k}_{2}}$$

$$k_{ij}^{l}$$
：第l层的卷积核的第i，j个元素

$${ {b}^{l}}$$
：第l层的偏置（特征图共用一个偏置）

$$x_{ij}^{l}$$
：第l层的第i，j个特征

$$o_{ij}^{l}$$
：第l层的第i，j个特征通过激活函数后的特征

$$f(\centerdot )$$
：激活函数

L：网络总层数

C：网络的损失函数

可知：

$$x_{ij}^{l}=\sum\limits_{p=0}^{ { {k}_{1}}-1}{\sum\limits_{q=0}^{ { {k}_{2}}-1}{o_{i+p,j+q}^{l-1}k_{pq}^{l}}}+{ {b}^{l}} \tag{1}$$

$$o_{ij}^{l}=f\left( x_{ij}^{l} \right) \tag{2}$$


## P2. 损失函数对k求导公式

$$\frac{\partial C}{\partial k_{pq}^{l}}=\sum\limits_{i=0}^{M-{ {k}_{1}}}{\sum\limits_{j=0}^{N-{ {k}_{2}}}{\frac{\partial C}{\partial x_{ij}^{l}}\frac{\partial x_{ij}^{l}}{\partial k_{pq}^{l}}}} \tag{3}$$

公式(3)中第二项：

$$\frac{\partial x_{ij}^{l}}{\partial k_{pq}^{l}}\text{=}\frac{\partial }{\partial k_{pq}^{l}}\left( \sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{x_{i+m,j+n}^{l-1}k_{mn}^{l}}}+{ {b}^{l}} \right)\xrightarrow{\begin{smallmatrix} 
 \text{仅当}m=p\text{且}n=q\text{时，}k_{mn}^{l}\text{与}k_{pq}^{l}\text{相关} \\ 
 { {b}^{l}}\text{与}k_{pq}^{l}\text{无关}
\end{smallmatrix}}x_{i+p,j+q}^{l-1} \tag{4}$$


## P3. 损失函数对b求导公式

$$\frac{\partial C}{\partial { {b}^{l}}}=\sum\limits_{i=0}^{M-{ {k}_{1}}}{\sum\limits_{j=0}^{N-{ {k}_{2}}}{\frac{\partial C}{\partial x_{ij}^{l}}\frac{\partial x_{ij}^{l}}{\partial { {b}^{l}}}}} \tag{5}$$

公式(5)中第二项

$$\frac{\partial x_{ij}^{l}}{\partial { {b}^{l}}}\text{=}\frac{\partial }{\partial { {b}^{l}}}\left( \sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{x_{i+m,j+n}^{l-1}k_{mn}^{l}}}+{ {b}^{l}} \right)\xrightarrow{\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{x_{i+m,j+n}^{l-1}k_{mn}^{l}}} \text{与}{ {b}^{l}}\text{无关}}1 \tag{6}$$


## P4. 损失函数对k、b求导结果

令反向传播时，损失函数在当前位置的误差

$$\delta _{ij}^{l}=\frac{\partial C}{\partial x_{ij}^{l}} \tag{7}$$

将公式(4)和公式(7)带入公式(3)，得损失函数对k的求导结果：

$$\frac{\partial C}{\partial k_{pq}^{l}}=\sum\limits_{i=0}^{M-{ {k}_{1}}}{\sum\limits_{j=0}^{N-{ {k}_{2}}}{\delta _{ij}^{l}x_{i+p,j+q}^{l-1}}} \tag{8}$$

将公式(6)和公式(7)带入公式(5)，得损失函数对b的求导结果：

$$\frac{\partial C}{\partial { {b}^{l}}}=\sum\limits_{i=0}^{M-{ {k}_{1}}}{\sum\limits_{j=0}^{N-{ {k}_{2}}}{\delta _{ij}^{l}}} \tag{9}$$


## P5. 具体计算反传时误差

假设已知损失在后一层的误差
$$\delta _{ij}^{l\text{+}1}$$
，则

$$\delta _{ij}^{l}=\frac{\partial C}{\partial x_{ij}^{l}}\text{=}\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{\frac{\partial C}{\partial x_{i-m,j-n}^{l+1}}\frac{\partial x_{i-m,j-n}^{l+1}}{\partial x_{ij}^{l}}}}\text{=}\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{\delta _{i-m,j-n}^{l+1}\frac{\partial x_{i-m,j-n}^{l+1}}{\partial x_{ij}^{l}}}} \tag{10}$$

如下图所示。假定卷积核大小为
$${ {k}_{1}}\times { {k}_{2}}$$
，输入点为(i, j)，其值为x(i, j)，其对应卷积后的特征为y(i, j)。由于在cnn中计算卷积时，从 (i-(k1-1), j-(k2-1))到(i, j)均会使用到该点，i-(k1-1)+(k1-1)=i，j-(k2-1)+(k2-1)=j，即在y上从(i-(k1-1), j-(k2-1))到(i, j)的点均会收到x上(i, j)的影响。因而公式(20)在使用连式法则时，要使用
$$\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{\frac{\bullet }{\partial x_{i-m,j-n}^{l+1}}}}$$
的坐标范围，且要使用(i-m, j-n)。

![1](/assets/post/2021-07-09-backpropConv/1.png)
_图1_

由于

$$x_{i-m,j-n}^{l+1}\text{=}\sum\limits_{s=0}^{ { {k}_{1}}-1}{\sum\limits_{t=0}^{ { {k}_{2}}-1}{o_{i-m+s,j-n+t}^{l}k_{st}^{l+1}}}+{ {b}^{l+1}} \tag{11}$$

公式(11)带入公式(10)第二项：

$$\begin{align}
  & \frac{\partial x_{i-m,j-n}^{l+1}}{\partial x_{ij}^{l}}\text{=}\frac{\partial }{\partial x_{ij}^{l}}\left( \sum\limits_{s=0}^{ { {k}_{1}}-1}{\sum\limits_{t=0}^{ { {k}_{2}}-1}{o_{i-m+s,j-n+t}^{l}k_{st}^{l+1}}}+{ {b}^{l+1}} \right) \\ 
 & \text{=}\frac{\partial }{\partial x_{ij}^{l}}\left( \sum\limits_{s=0}^{ { {k}_{1}}-1}{\sum\limits_{t=0}^{ { {k}_{2}}-1}{f\left( x_{i-m+s,j-n+t}^{l} \right)k_{st}^{l+1}}}+{ {b}^{l+1}} \right) \\ 
 & \xrightarrow{\text{只有}s=m \text{，} t=n\text{时，}f\left( x_{i-m+s,j-n+t}^{l} \right)\text{与} x_{ij}^{l} \text{有关} }\frac{\partial \left( k_{mn}^{l+1}f\left( x_{ij}^{l} \right) \right)}{\partial x_{ij}^{l}}=k_{mn}^{l+1}{ {f}^{'}}\left( x_{ij}^{l} \right) \\ 
\end{align}$$

即：

$$\frac{\partial x_{i-m,j-n}^{l+1}}{\partial x_{ij}^{l}}=k_{mn}^{l+1}{ {f}^{'}}\left( x_{ij}^{l} \right) \tag{12}$$

其中
$${ {f}^{'}}\left( x_{ij}^{l} \right)$$
为激活函数的导数。

公式(12)带入公式(10)：

$$\begin{align}
  & \delta _{ij}^{l}\text{=}\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{\delta _{i-m,j-n}^{l+1}k_{mn}^{l+1}{ {f}^{'}}\left( x_{ij}^{l} \right)}} \\ 
 & =ro{ {t}_{ { {180}^{\circ }}}}\left\{ \sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{\delta _{i+m,j+n}^{l+1}k_{mn}^{l+1}}} \right\}{ {f}^{'}}\left( x_{ij}^{l} \right) \\ 
 & =conv2D\left( \delta _{i,j}^{l+1},ro{ {t}_{ { {180}^{\circ }}}}\left\{ { {k}^{l+1}} \right\} \right){ {f}^{'}}\left( x_{ij}^{l} \right) \\ 
\end{align} \tag{13}$$

转换成矩阵形式：

$${ {\mathbf{\delta }}^{l}}=conv2D\left( { {\mathbf{\delta }}^{l+1}},ro{ {t}_{ { {180}^{\circ }}}}\left\{ { {\mathbf{k}}^{l+1}} \right\} \right)\odot { {f}^{'}}\left( { {\mathbf{x}}^{l}} \right) \tag{14}$$

其中
$$\odot $$
代表对应元素相乘。即卷积层反向传播时，本层误差，是‘后一层误差’和‘上下左右旋转后的后一层卷积核’相卷积，再和‘当前层激活函数的导数’对应元素相乘后的结果。


## P6. 补充知识

给定图像
$$I$$
和卷积核
$$K\in { {R}^{ { {k}_{1}}\times { {k}_{2}}}}$$

**互相关**

$${ {\left( I\otimes K \right)}_{ij}}=\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{I\left( i+m,j+n \right)K\left( m,n \right)}} \tag{15}$$

**卷积**

$$\begin{align}
  & { {\left( I*K \right)}_{ij}}=\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{I\left( i-m,j-n \right)K\left( m,n \right)}} \\ 
 & \text{             }=\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{I\left( i+m,j+n \right)K\left( -m,-n \right)}}\text{ } \\ 
\end{align} \tag{16}$$

可见，将卷积核上下左右镜像（上下镜像：-m，左右镜像：-n）之后，卷积和互相关一样。

另外，神经网络中使用的卷积，实际上是互相关，如下（b为偏置）：

$${ {\left( I*K \right)}_{ij}}=\sum\limits_{m=0}^{ { {k}_{1}}-1}{\sum\limits_{n=0}^{ { {k}_{2}}-1}{I\left( i+m,j+n \right)K\left( m,n \right)}}\text{+}b \tag{17}$$

**rot180**

如下图，将图像上下左右镜像之后，和将图像直接旋转180°，结果是一样的。

![2](/assets/post/2021-07-09-backpropConv/2.png)
_图2_
