---
layout: post
title:  "全连接层梯度反向传播的推导"
date:   2021-07-02 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/backpropFC>


令
$$X\in { {R}^{M\times D}}$$
，
$$W\in { {R}^{D\times N}}$$
，全连接层
$$Y=XW\in { {R}^{M\times N}}$$
，其中
$${ {y}_{ij}}=\sum\limits_{k=1}^{D}{ { {x}_{ik}}\centerdot { {w}_{kj}}}$$
，如下图所示。

![1](/assets/post/2021-07-02-backpropFC/1backpropFC.png)
_图1_

其中

$$X=\left[ \begin{matrix}
   { {x}_{11}} & \cdots  & { {x}_{1D}}  \\
   \vdots  & { {x}_{ij}} & \vdots   \\
   { {x}_{M1}} & \cdots  & { {x}_{MD}}  \\
\end{matrix} \right]$$，$$W=\left[ \begin{matrix}
   { {w}_{11}} & \cdots  & { {w}_{1N}}  \\
   \vdots  & { {w}_{ij}} & \vdots   \\
   { {w}_{D1}} & \cdots  & { {w}_{DN}}  \\
\end{matrix} \right]$$

令损失为L（标量），则：

$$\frac{\partial L}{\partial Y}=\left[ \begin{matrix}
   \frac{\partial L}{\partial { {y}_{11}}} & \cdots  & \frac{\partial L}{\partial { {y}_{1N}}}  \\
   \vdots  & \frac{\partial L}{\partial { {y}_{ij}}} & \vdots   \\
   \frac{\partial L}{\partial { {y}_{M1}}} & \cdots  & \frac{\partial L}{\partial { {y}_{MN}}}  \\
\end{matrix} \right]$$

$${ {X}^{T}}=\left[ \begin{matrix}
   { {x}_{11}} & \cdots  & { {x}_{M1}}  \\
   \vdots  & { {x}_{ji}} & \vdots   \\
   { {x}_{1D}} & \cdots  & { {x}_{MD}}  \\
\end{matrix} \right]$$

$${ {W}^{T}}=\left[ \begin{matrix}
   { {w}_{11}} & \cdots  & { {w}_{D1}}  \\
   \vdots  & { {w}_{ji}} & \vdots   \\
   { {w}_{1N}} & \cdots  & { {w}_{DN}}  \\
\end{matrix} \right]$$

**1. L关于X的偏导数**

① Y关于
$$x_{ij}$$
的偏导数：

$$\frac{\partial Y}{\partial { {x}_{ij}}}=\left[ \begin{matrix}
   \frac{\partial { {y}_{11}}}{\partial { {x}_{ij}}} & ... & \frac{\partial { {y}_{1N}}}{\partial { {x}_{ij}}}  \\
   \vdots  & \frac{\partial { {y}_{mn}}}{\partial { {x}_{ij}}} & \vdots   \\
   \frac{\partial { {y}_{M1}}}{\partial { {x}_{ij}}} & ... & \frac{\partial { {y}_{MN}}}{\partial { {x}_{ij}}}  \\
\end{matrix} \right]\xrightarrow[\begin{smallmatrix} 
 \text{即}Y\text{的第}i\text{行只与}X\text{的第}i\text{行有关} \\ 
 \text{（}Y\text{的第}j\text{列只与}W\text{的第}j\text{列有关} 
\end{smallmatrix}]{\text{由上图，只有当}m\text{=}i\text{时，}Y\text{才与}{ {x}_{ij}}\text{有关}}=\left[ \begin{matrix}
   0 & ... & 0  \\
   \frac{\partial { {y}_{i0}}}{\partial { {x}_{ij}}} & \frac{\partial { {y}_{in}}}{\partial { {x}_{ij}}} & \frac{\partial { {y}_{iN}}}{\partial { {x}_{ij}}}  \\
   0 & ... & 0  \\
\end{matrix} \right]$$

其中
$$1\le i\le M$$
，
$$1\le j\le D$$
，
$$1\le m\le M$$
，
$$1\le n\le N$$。

② 接下来计算：

$$\begin{align}
  & \frac{\partial { {y}_{in}}}{\partial { {x}_{ij}}}\text{=}\frac{\partial \sum\limits_{k=1}^{D}{ { {x}_{ik}}\centerdot { {w}_{kn}}}}{\partial { {x}_{ij}}}=\sum\limits_{k=1}^{D}{\frac{\partial { {x}_{ik}}}{\partial { {x}_{ij}}}\centerdot { {w}_{kn}}} \\ 
 & \text{      }=\frac{\partial { {x}_{i1}}}{\partial { {x}_{ij}}}\centerdot { {w}_{1n}}+\cdots +\frac{\partial { {x}_{ij}}}{\partial { {x}_{ij}}}\centerdot { {w}_{jn}}+\cdots +\frac{\partial { {x}_{iD}}}{\partial { {x}_{ij}}}\centerdot { {w}_{Dn}} \\ 
 & \text{      }=0+\cdots +{ {w}_{jn}}+\cdots +0={ {w}_{jn}} \\ 
\end{align}$$

因而
$$\frac{\partial Y}{\partial { {x}_{ij}}}$$
只有第i行不为0，其他行均为0：

$$\frac{\partial Y}{\partial { {x}_{ij}}}=\left[ \begin{matrix}
   0 & ... & 0  \\
   { {w}_{j1}} & { {w}_{jn}} & { {w}_{jN}}  \\
   0 & ... & 0  \\
\end{matrix} \right]\text{  }\begin{matrix}
  \text{第}0\text{行} \\
  \text{第}i\text{行} \\
  \text{第}M\text{行}  \\
\end{matrix}$$

③ 根据链式法则，

$$\frac{\partial L}{\partial { {x}_{ij}}}=\sum\limits_{m=1,n=1}^{M,N}{\frac{\partial L}{\partial { {y}_{mn}}}\frac{\partial { {y}_{mn}}}{\partial { {x}_{ij}}}}\xrightarrow{m=i\text{时，}\frac{\partial { {y}_{mn}}}{\partial { {x}_{ij}}}\ne 0}\sum\limits_{n=1}^{N}{\frac{\partial L}{\partial { {y}_{in}}}\frac{\partial { {y}_{in}}}{\partial { {x}_{ij}}}}$$

由于
$$\frac{\partial { {y}_{in}}}{\partial { {x}_{ij}}}\text{=}{ {w}_{jn}}$$
，因而

$$\begin{align}
  & \frac{\partial L}{\partial { {x}_{ij}}}=\sum\limits_{n=1}^{N}{\frac{\partial L}{\partial { {y}_{in}}}{ {w}_{jn}}}\xrightarrow{n\text{换成}k\text{,不影响结果}}\sum\limits_{\text{k}=1}^{N}{\frac{\partial L}{\partial { {y}_{ik}}}{ {w}_{jk}}} \\ 
 & \text{       =}\frac{\partial L}{\partial { {y}_{i1}}}{ {w}_{j1}}+\cdots +\frac{\partial L}{\partial { {y}_{ik}}}{ {w}_{jk}}+\cdots +\frac{\partial L}{\partial { {y}_{iN}}}{ {w}_{jN}} \\ 
 & \text{      }=\left[ \begin{matrix}
   \frac{\partial L}{\partial { {y}_{i1}}} & \cdots  & \frac{\partial L}{\partial { {y}_{ik}}} & \cdots  & \frac{\partial L}{\partial { {y}_{iN}}}  \\
\end{matrix} \right]\left[ \begin{matrix}
   { {w}_{j1}}  \\
   \vdots   \\
   { {w}_{jk}}  \\
   \vdots   \\
   { {w}_{jN}}  \\
\end{matrix} \right] \\ 
\end{align}$$

即

$$\frac{\partial L}{\partial { {x}_{ij}}}\text{=}\sum\limits_{\text{k}=1}^{N}{\frac{\partial L}{\partial { {y}_{ik}}}{ {w}_{jk}}}$$

也即L关于X的第i行第j列的元素xij的偏导数，是L关于Y的第i行的偏导数和WT的第j列的元素的点乘（对应元素相乘并相加）的结果。

④ 写成矩阵形式：

$$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y}{ {W}^{T}}$$


**2. L关于W的偏导数**

① Y关于wij的偏导数：

$$\frac{\partial Y}{\partial { {w}_{ij}}}=\left[ \begin{matrix}
   \frac{\partial { {y}_{11}}}{\partial { {w}_{ij}}} & ... & \frac{\partial { {y}_{1N}}}{\partial { {w}_{ij}}}  \\
   \vdots  & \frac{\partial { {y}_{mn}}}{\partial { {w}_{ij}}} & \vdots   \\
   \frac{\partial { {y}_{M1}}}{\partial { {w}_{ij}}} & ... & \frac{\partial { {y}_{MN}}}{\partial { {w}_{ij}}}  \\
\end{matrix} \right]\xrightarrow[\begin{smallmatrix} 
 \text{即}Y\text{的第}j\text{列只与}W\text{的第}j\text{列有关} \\ 
 \text{（}Y\text{的第}i\text{行只与}X\text{的第}i\text{行有关）} 
\end{smallmatrix}]{\text{由上图，只有当}n\text{=}j\text{时，}Y\text{才与}{ {w}_{ij}}\text{有关}}=\left[ \begin{matrix}
   0 & \frac{\partial { {y}_{1j}}}{\partial { {w}_{ij}}} & 0  \\
   \vdots  & \frac{\partial { {y}_{mj}}}{\partial { {w}_{ij}}} & \vdots   \\
   0 & \frac{\partial { {y}_{Mj}}}{\partial { {w}_{ij}}} & 0  \\
\end{matrix} \right]$$

其中
$$1\le i\le M$$
，
$$1\le j\le D$$
，
$$1\le m\le M$$
，
$$1\le n\le N$$
。

② 接下来计算：

$$\begin{align}
  & \frac{\partial { {y}_{mj}}}{\partial { {w}_{ij}}}\text{=}\frac{\partial \sum\limits_{k=1}^{D}{ { {x}_{mk}}\centerdot { {w}_{kj}}}}{\partial { {w}_{ij}}}=\sum\limits_{k=1}^{D}{ { {x}_{mk}}\centerdot \frac{\partial { {w}_{kj}}}{\partial { {w}_{ij}}}} \\ 
 & \text{      }={ {x}_{m1}}\centerdot \frac{\partial { {w}_{1j}}}{\partial { {w}_{ij}}}+\cdots +{ {x}_{mi}}\centerdot \frac{\partial { {w}_{ij}}}{\partial { {w}_{ij}}}+\cdots +{ {x}_{mD}}\centerdot \frac{\partial { {w}_{Dj}}}{\partial { {w}_{ij}}} \\ 
 & \text{      }=0+\cdots +{ {x}_{mi}}+\cdots +0={ {x}_{mi}} \\ 
\end{align}$$


因而
$$\frac{\partial Y}{\partial { {w}_{ij}}}$$
只有第j列不为0，其他列均为0：

$$\frac{\partial Y}{\partial { {x}_{ij}}}=\begin{matrix}
   \left[ \begin{matrix}
   0 & { {w}_{1i}} & 0  \\
   \vdots  & { {w}_{mi}} & \vdots   \\
   0 & { {w}_{Mi}} & 0  \\
\end{matrix} \right]  \\
   \begin{matrix}
   \text{第}0\text{列} & \text{第}j\text{列} & \text{第}M\text{列}  \\
\end{matrix}  \\
\end{matrix}$$

③ 根据链式法则，

$$\frac{\partial L}{\partial { {w}_{ij}}}=\sum\limits_{m=1,n=1}^{M,N}{\frac{\partial L}{\partial { {y}_{mn}}}\frac{\partial { {y}_{mn}}}{\partial { {w}_{ij}}}}\xrightarrow{n=j\text{时，}\frac{\partial { {y}_{mn}}}{\partial { {w}_{ij}}}\ne 0}\sum\limits_{m=1}^{M}{\frac{\partial L}{\partial { {y}_{mj}}}\frac{\partial { {y}_{mj}}}{\partial { {w}_{ij}}}}$$

由于
$$\frac{\partial { {y}_{mj}}}{\partial { {w}_{ij}}}\text{=}{ {x}_{mi}}$$
，因而

$$\begin{align}
  & \frac{\partial L}{\partial { {w}_{ij}}}=\sum\limits_{m=1}^{M}{\frac{\partial L}{\partial { {y}_{mj}}}{ {x}_{mi}}}\xrightarrow{m\text{换成}k\text{，不影响结果}}\sum\limits_{k=1}^{M}{\frac{\partial L}{\partial { {y}_{kj}}}{ {x}_{ki}}}=\sum\limits_{k=1}^{M}{ { {x}_{ki}}\frac{\partial L}{\partial { {y}_{kj}}}} \\ 
 & \text{       =}{ {x}_{1i}}\frac{\partial L}{\partial { {y}_{1j}}}+\cdots +{ {x}_{ki}}\frac{\partial L}{\partial { {y}_{kj}}}+\cdots +{ {x}_{Mi}}\frac{\partial L}{\partial { {y}_{Mj}}} \\ 
 & \text{      }=\left[ \begin{matrix}
   { {x}_{1i}} & \cdots  & { {x}_{ki}} & \cdots  & { {x}_{Mi}}  \\
\end{matrix} \right]\left[ \begin{matrix}
   \frac{\partial L}{\partial { {y}_{1j}}}  \\
   \vdots   \\
   \frac{\partial L}{\partial { {y}_{kj}}}  \\
   \vdots   \\
   \frac{\partial L}{\partial { {y}_{Mj}}}  \\
\end{matrix} \right] \\ 
\end{align}$$

即

$$\frac{\partial L}{\partial { {w}_{ij}}}=\sum\limits_{k=1}^{M}{ { {x}_{ki}}\frac{\partial L}{\partial { {y}_{kj}}}}$$

也即L关于W的第i行第j列的元素wij的偏导数，是
$${ {x}^{T}}$$
的第i行的元素和L关于Y的第j列的偏导数的点乘（对应元素相乘并相加）的结果。

④ 写成矩阵形式：

$$\frac{\partial L}{\partial W}={ {X}^{T}}\bullet \frac{\partial L}{\partial Y}$$
