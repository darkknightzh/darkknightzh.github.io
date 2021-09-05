---
layout: post
title:  "神经网络反向传播公式的推导"
date:   2021-07-08 16:00:00 +0800
tags: [deep learning, algorithm]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/backpropNN>


## P1. 变量定义
$$w_{ij}^{k}$$
：第k层的第i个神经元和第k-1层的第j个神经元之间的权重

$$b_{i}^{k}$$
：第k层的第i个神经元的偏置

$$z_{i}^{k}$$
：第k层的第i个神经元的输入

$$a_{i}^{k}$$
：第k层的第i个神经元的输出，且
$$a_{i}^{k}=\sigma \left( z_{i}^{k} \right)$$

$${ {n}^{k}}$$
：第k层的神经元个数

$$\sigma $$
：隐含层的激活函数

L：网络总层数（最后一层为输出层，其输入为
$$a_{i}^{L}$$
）

可知第k层的第i个神经元的输入和输出：

$$z_{i}^{k}=\sum\limits_{j=1}^{ { {n}^{k\text{-}1}}}{w_{ij}^{k}a_{j}^{k-1}}+b_{i}^{k} \tag{1}$$

其中
$$\sum\limits_{j=1}^{ { {n}^{k\text{-}1}}}{w_{ij}^{k}a_{j}^{k-1}}$$
是所有的“&nbsp;'第k层的第i个神经元和第k-1层的第j个神经元之间的权重'与'第k-1层的第j个神经元的输出'的乘积”之和。
$$b_{i}^{k}$$
代表第k层的第i个神经元的输入（
$$z_{i}^{k}$$
）只有一个偏置。

损失函数为均方误差（不具体指定损失函数形式时，记为
$$C(X,\theta )$$
）：

$$C(X,\theta )=\frac{1}{2N}\sum\limits_{d=1}^{N}{ { {\left( a_{d}^{L}-{ {y}_{d}} \right)}^{2}}}\tag{2}$$ 

其中
$${ {y}_{d}}$$
为对于第d个输入
$${ {x}_{d}}$$
的目标值，
$$a_{d}^{L}$$
为第d个输入
$${ {x}_{d}}$$
通过神经网络得到的实际值。
$$N={ {n}^{L}}$$
为输出层节神经元个数。 

因而令
$$C=\frac{1}{2}{ {\left( { {a}^{L}}-y \right)}^{2}}$$
代表输出层第d个输入对应的误差。为方便显示，此处省略下标d。


## P2. 损失函数通用求导公式

**① 损失函数对w求导**

$$\frac{\partial C(X,\theta )}{\partial w_{ij}^{k}}=\frac{1}{N}\sum\limits_{d=1}^{N}{\frac{\partial \frac{1}{2}{ {\left( a_{d}^{L}-{ {y}_{d}} \right)}^{2}}}{\partial w_{ij}^{k}}}\xrightarrow{ { {C}_{d}}\text{=}\frac{1}{2}{ {\left( a_{d}^{L}-{ {y}_{d}} \right)}^{2}}}\frac{1}{N}\sum\limits_{d=1}^{N}{\frac{\partial { {C}_{d}}}{\partial w_{ij}^{k}}} \tag{3}$$

即需要求第d个输入对应的误差
$$C=\frac{1}{2}{ {\left( { {a}^{L}}-y \right)}^{2}}$$
对
$$w_{ij}^{k}$$
的偏导数
$$\frac{\partial C}{\partial w_{ij}^{k}}$$
，此处省略下标d。

$$\frac{\partial C}{\partial w_{ij}^{k}}=\frac{\partial C}{\partial z_{i}^{k}}\frac{\partial z_{i}^{k}}{\partial w_{ij}^{k}} \tag{4}$$

由于
$$w_{ij}^{k}$$
连接的是第k层的第i个神经元，因而应用链式求导时，需要使用
$$\partial z_{i}^{k}$$
。公式(4)第二项为：

$$\frac{\partial z_{i}^{k}}{\partial w_{ij}^{k}}=\frac{\partial \left( \sum\limits_{l=1}^{ { {n}^{k\text{-}1}}}{w_{il}^{k}a_{l}^{k-1}}+b_{i}^{k} \right)}{\partial w_{ij}^{k}}=\xrightarrow{\begin{smallmatrix} 
 \text{仅当} l=j\text{时，}w_{il}^{k} \text{与} w_{ij}^{k} \text{有关}  \\ 
 b_{i}^{k} \text{与} w_{ij}^{k} \text{无关} 
\end{smallmatrix}}a_{j}^{k-1} \tag{5}$$

令公式(4)第一项记为

$$\delta _{i}^{k}=\frac{\partial C}{\partial z_{i}^{k}} \tag{6}$$

代表反向传播时第k层的第i个神经元的误差，并将公式(5)、公式(6)带入公式(4)，得：

$$\frac{\partial C}{\partial w_{ij}^{k}}=\delta _{i}^{k}a_{j}^{k-1} \tag{7}$$

即C对
$$w_{ij}^{k}$$
的偏导，为第k层第i个神经元的误差
$$\delta _{i}^{k}$$
和第k-1层第j个神经元的输出
$$a_{j}^{k-1}$$
的乘积。直觉上，
$$w_{ij}^{k}$$
连接第k层第i个神经元和第k-1层第j个神经元（
$$a_{j}^{k-1}$$
），和公式吻合。

**② 损失函数对b求导**

$$\frac{\partial C(X,\theta )}{\partial b_{i}^{k}}=\frac{1}{N}\sum\limits_{d=1}^{N}{\frac{\partial \frac{1}{2}{ {\left( a_{d}^{L}-{ {y}_{d}} \right)}^{2}}}{\partial b_{i}^{k}}}\xrightarrow{ { {C}_{d}}\text{=}\frac{1}{2}{ {\left( a_{d}^{L}-{ {y}_{d}} \right)}^{2}}}\frac{1}{N}\sum\limits_{d=1}^{N}{\frac{\partial { {C}_{d}}}{\partial b_{i}^{k}}} \tag{8}$$

即需要求第d个输入对应的误差
$$C=\frac{1}{2}{ {\left( \hat{y}-y \right)}^{2}}$$
对
$$b_{i}^{k}$$
的偏导数
$$\frac{\partial C}{\partial b_{i}^{k}}$$
，此处省略下标d。下面计算：

$$\frac{\partial C}{\partial b_{i}^{k}}=\frac{\partial C}{\partial z_{i}^{k}}\frac{\partial z_{i}^{k}}{\partial b_{i}^{k}} \tag{9}$$

此处
$$b_{i}^{k}$$
为
$$z_{i}^{k}$$
的偏置，因而应用链式求导时，需要使用
$$\partial z_{i}^{k}$$
。

公式(9)第二项

$$\frac{\partial z_{i}^{k}}{\partial b_{i}^{k}}=\frac{\partial \left( \sum\limits_{l=1}^{ { {n}^{k\text{-}1}}}{w_{il}^{k}a_{l}^{k-1}}+b_{i}^{k} \right)}{\partial b_{i}^{k}}=\xrightarrow{\sum\limits_{l=1}^{ { {n}^{k\text{-}1}}}{w_{il}^{k}a_{l}^{k-1}}\text{和}b_{i}^{k} \text{无关}}1 \tag{10}$$

将公式(6)和公式(10)带入公式(9)：

$$\frac{\partial C}{\partial b_{i}^{k}}=\frac{\partial C}{\partial z_{i}^{k}}\frac{\partial z_{i}^{k}}{\partial b_{i}^{k}}\text{=}\delta _{i}^{k}\centerdot 1\text{=}\delta _{i}^{k}$$

即：

$$\frac{\partial C}{\partial b_{i}^{k}}\text{=}\delta _{i}^{k} \tag{11}$$

可见损失对
$$b_{i}^{k}$$
的偏导只和为第k层第i个神经元的误差
$$\delta _{i}^{k}$$
有关。因而下面对输出层或隐含层计算该偏导时，不再计算。


## P3. 输出层对w求导公式

当时用均方误差计算损失时

$${ {C}_{d}}=\frac{1}{2}{ {\left( a_{d}^{L}-{ {y}_{d}} \right)}^{2}}=\frac{1}{2}{ {\left( \sigma \left( z_{d}^{L} \right)-{ {y}_{d}} \right)}^{2}}$$

进而

$$\delta _{d}^{L}=\frac{\partial { {C}_{d}}}{\partial z_{d}^{k}}=\frac{\partial \left( \frac{1}{2}{ {\left( \sigma \left( z_{d}^{L} \right)-{ {y}_{d}} \right)}^{2}} \right)}{\partial z_{d}^{k}}=\left( \sigma \left( z_{d}^{L} \right)-y \right)\centerdot { {\sigma }^{'}}\left( z_{d}^{L} \right)=\left( a_{d}^{L}-{ {y}_{d}} \right)\centerdot { {\sigma }^{'}}\left( z_{d}^{L} \right) \tag{12}$$

故

$$\frac{\partial C}{\partial w_{ij}^{L}}=\delta _{i}^{L}a_{j}^{k-1}\text{=}\left( a_{d}^{L}-{ {y}_{i}} \right)\centerdot { {\sigma }^{'}}\left( z_{i}^{L} \right)a_{j}^{L-1} \tag{13}$$

注：若此处不限定
$$C(X,\theta )$$
的具体形式（第d个输入的损失用
$${ {C}_{d}}$$
表示），则公式(12)和公式(13)的形式为：

$$\delta _{d}^{L}=\frac{\partial { {C}_{d}}}{\partial z_{d}^{k}}=\frac{\partial { {C}_{d}}}{\partial a_{d}^{L}}\frac{\partial a_{d}^{L}}{\partial z_{d}^{L}}={ {\nabla }_{a}}C\centerdot { {\sigma }^{'}}\left( z_{d}^{L} \right) \tag{14}$$

$$\frac{\partial C}{\partial w_{ij}^{L}}=\delta _{i}^{L}a_{j}^{L-1}\text{=}{ {\nabla }_{a}}C\centerdot { {\sigma }^{'}}\left( z_{d}^{L} \right)a_{j}^{L-1} \tag{15}$$

其中
$${ {\nabla }_{a}}C$$
代表损失函数C对输入变量a的偏导数。


## P4. 隐含层对w求导公式

$$\delta _{i}^{k}=\frac{\partial C}{\partial z_{i}^{k}}\text{=}\sum\limits_{l=1}^{ { {n}^{k+1}}}{\frac{\partial C}{\partial z_{l}^{k+1}}\centerdot \frac{\partial z_{l}^{k+1}}{\partial z_{i}^{k}}}\text{=}\sum\limits_{l=1}^{ { {n}^{k+1}}}{\delta _{l}^{k+1}\centerdot \frac{\partial z_{l}^{k+1}}{\partial z_{i}^{k}}} \tag{16}$$

此处
$${ {n}^{k+1}}$$
为k+1层神经元个数。由此可见，第k层（非最后一层）第i个神经元的误差
$$\delta _{i}^{k}$$
依赖于k+1层的所有误差。因而误差从输出流向输入。

由于
$$z_{l}^{k+1}=\sum\limits_{j=1}^{ { {n}^{k}}}{w_{lj}^{k+1}a_{j}^{k}}+b_{l}^{k+1}=\sum\limits_{j=1}^{ { {n}^{k}}}{w_{lj}^{k+1}\sigma \left( z_{j}^{k} \right)}+b_{l}^{k+1} \tag{17}$$ 

故公式(16)第二项

$$\frac{\partial z_{l}^{k+1}}{\partial z_{i}^{k}}\text{=}\frac{\sum\limits_{j=1}^{ { {n}^{k}}}{w_{lj}^{k+1}\sigma \left( z_{j}^{k} \right)}+b_{l}^{k+1}}{\partial z_{i}^{k}}\xrightarrow{\begin{smallmatrix} 
 j=i\text{时，}z_{j}^{k} \text{与} z_{i}^{k} \text{有关} \\ 
 b_{l}^{k+1} \text{与} z_{i}^{k} \text{无关} 
\end{smallmatrix}}w_{li}^{k+1}\centerdot { {\sigma }^{'}}\left( z_{i}^{k} \right) \tag{18}$$

公式(18)带入公式(16)：

$$\delta _{i}^{k}=\sum\limits_{l=1}^{ { {n}^{k+1}}}{\delta _{l}^{k+1}\centerdot w_{li}^{k+1}\centerdot { {\sigma }^{'}}\left( z_{i}^{k} \right)}\text{=}{ {\sigma }^{'}}\left( z_{i}^{k} \right)\centerdot \sum\limits_{l=1}^{ { {n}^{k+1}}}{\delta _{l}^{k+1}\centerdot w_{li}^{k+1}} \tag{19}$$

将公式(19)带入公式(7)：

$$\frac{\partial C}{\partial w_{ij}^{k}}=\delta _{i}^{k}a_{j}^{k-1}\text{=}{ {\sigma }^{'}}\left( z_{i}^{k} \right)\centerdot a_{j}^{k-1}\centerdot \sum\limits_{l=1}^{ { {n}^{k+1}}}{\delta _{l}^{k+1}\centerdot w_{li}^{k+1}} \tag{20}$$


## P5. 矩阵形式

令
$${ {\mathbf{z}}^{k}}\text{=}{ {\left[ z_{1}^{k},\cdots ,z_{i}^{k},\cdots ,z_{ { {n}^{k}}}^{k} \right]}^{T}}$$
，
$${ {\mathbf{a}}^{k}}\text{=}{ {\left[ a_{1}^{k},\cdots ,a_{i}^{k},\cdots ,a_{ { {n}^{k}}}^{k} \right]}^{T}}$$
，
$${ {\mathbf{b}}^{k}}\text{=}{ {\left[ b_{1}^{k},\cdots ,b_{i}^{k},\cdots ,b_{ { {n}^{k}}}^{k} \right]}^{T}}$$
，
$${ {\mathbf{\delta }}^{k}}={ {\left[ \delta _{1}^{k},\cdots ,\delta _{i}^{k},\cdots ,\delta _{ { {n}^{k}}}^{k} \right]}^{T}}$$
，
$$\frac{\partial C}{\partial { {\mathbf{b}}^{k}}}={ {\left[ \frac{\partial C}{\partial b_{1}^{k}},\cdots ,\frac{\partial C}{\partial b_{i}^{k}},\cdots ,\frac{\partial C}{\partial b_{ { {n}^{k}}}^{k}} \right]}^{T}}$$
，
$${ {\sigma }^{'}}\left( { {\mathbf{z}}^{k}} \right)={ {\left[ \frac{\partial \sigma \left( z_{1}^{k} \right)}{\partial z_{1}^{k}},\cdots ,\frac{\partial \sigma \left( z_{i}^{k} \right)}{\partial z_{i}^{k}},\cdots ,\frac{\partial \sigma \left( z_{ { {n}^{k}}}^{k} \right)}{\partial z_{ { {n}^{k}}}^{k}} \right]}^{T}}$$

$${ {\mathbf{w}}^{k}}=\left[ \begin{matrix}
   w_{11}^{k} & \cdots  & w_{1{ {n}^{k-1}}}^{k}  \\
   \vdots  & w_{ij}^{k} & \vdots   \\
   w_{ { {n}^{k}}1}^{k} & \cdots  & w_{ { {n}^{k}}{ {n}^{k-1}}}^{k}  \\
\end{matrix} \right]$$，$$\frac{\partial C}{\partial { {\mathbf{w}}^{k}}}=\left[ \begin{matrix}
   \frac{\partial C}{\partial w_{11}^{k}} & \cdots  & \frac{\partial C}{\partial w_{1{ {n}^{k-1}}}^{k}}  \\
   \vdots  & \frac{\partial C}{\partial w_{ij}^{k}} & \vdots   \\
   \frac{\partial C}{\partial w_{ { {n}^{k}}1}^{k}} & \cdots  & \frac{\partial C}{\partial w_{ { {n}^{k}}{ {n}^{k-1}}}^{k}}  \\
\end{matrix} \right]$$

公式(1)：

$${ {\mathbf{z}}^{k}}={ {\mathbf{w}}^{k}}{ {\mathbf{a}}^{k-1}}+{ {\mathbf{b}}^{k}}$$

公式(2)：

$$C(X,\theta )=\frac{1}{2N}\sum\limits_{d=1}^{N}{ { {\left( a_{d}^{L}-{ {y}_{d}} \right)}^{2}}}=\frac{1}{2}\left\| { {\mathbf{a}}^{L}}-\mathbf{y} \right\|_{2}^{2}$$

公式(6)：

$${ {\mathbf{\delta }}^{k}}=\frac{\partial C}{\partial { {\mathbf{z}}^{k}}}$$

公式(7)：

$$\frac{\partial C}{\partial { {\mathbf{w}}^{k}}}={ {\mathbf{\delta }}^{k}}{ {\left( { {\mathbf{a}}^{k}} \right)}^{T}} \quad\quad\quad \mathbf{\color{red}{\text{权重w梯度的计算方法}}}$$

公式(11)：

$$\frac{\partial C}{\partial { {\mathbf{b}}^{k}}}={ {\mathbf{\delta }}^{k}} \quad\quad\quad \mathbf{\color{red}{\text{偏置b梯度的计算方法}}}$$

公式(12)：

$${ {\mathbf{\delta }}^{L}}=\left( { {\mathbf{a}}^{L}}-\mathbf{y} \right)\odot { {\sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)\text{=}{ {\Sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)\left( { {\mathbf{a}}^{L}}-\mathbf{y} \right)$$

其中
$${ {\Sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)$$
为对角矩阵，其对角线上的值为
$${ {\sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)$$
，
$$\odot $$
表示Hadamard乘积，代表对应元素相乘。

公式(13)：

$$\frac{\partial C}{\partial { {\mathbf{w}}^{L}}}\text{=}{ {\left( \left( { {\mathbf{a}}^{L}}-\mathbf{y} \right)\odot { {\sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right) \right)}^{T}}\centerdot { {\mathbf{a}}^{L-1}}\text{=}{ {\left( { {\Sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)\left( { {\mathbf{a}}^{L}}-\mathbf{y} \right) \right)}^{T}}\centerdot { {\mathbf{a}}^{L-1}}$$

公式(14)：

$${ {\mathbf{\delta }}^{L}}={ {\nabla }_{\mathbf{a}}}C\odot { {\sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)\text{=}{ {\Sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right){ {\nabla }_{\mathbf{a}}}C \quad\quad\quad \mathbf{\color{red}{\text{输出层误差}}}$$

其中
$${ {\nabla }_{\mathbf{a}}}C$$
代表损失函数C对输入向量a的偏导数向量

公式(15)：

$$\frac{\partial C}{\partial { {\mathbf{w}}^{L}}}={ {\left( { {\nabla }_{\mathbf{a}}}C\centerdot { {\sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right) \right)}^{T}}{ {\mathbf{a}}^{L-1}}$$

公式(19)：

$${ {\mathbf{\delta }}^{k}}=\left( { {\left( { {\mathbf{w}}^{k+1}} \right)}^{T}}{ {\mathbf{\delta }}^{k+1}} \right)\odot { {\sigma }^{'}}\left( { {\mathbf{z}}^{k}} \right)={ {\Sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)\left( { {\left( { {\mathbf{w}}^{k+1}} \right)}^{T}}{ {\mathbf{\delta }}^{k+1}} \right) \quad\quad\quad \mathbf{\color{red}{\text{中间层误差}}}$$      

公式(20)：

$$\frac{\partial E}{\partial { {\mathbf{w}}^{k}}}={ {\left( { {\mathbf{\delta }}^{k}} \right)}^{T}}{ {\mathbf{a}}^{k-1}}={ {\left( \left( { {\left( { {\mathbf{w}}^{k+1}} \right)}^{T}}{ {\mathbf{\delta }}^{k+1}} \right)\odot { {\sigma }^{'}}\left( { {\mathbf{z}}^{k}} \right) \right)}^{T}}{ {\mathbf{a}}^{k-1}}\text{=}{ {\left( { {\Sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)\left( { {\left( { {\mathbf{w}}^{k+1}} \right)}^{T}}{ {\mathbf{\delta }}^{k+1}} \right) \right)}^{T}}{ {\mathbf{a}}^{k-1}}$$


## P6. 反向传播流程

输入x：设置每层的激活函数
$$\sigma $$

前向传播：对于层数k=2,3,…,L，计算
$${ {\mathbf{z}}^{k}}={ {\mathbf{w}}^{k}}{ {\mathbf{a}}^{k-1}}+{ {\mathbf{b}}^{k}}$$
，
$${ {\mathbf{a}}^{k}}=\sigma \left( { {\mathbf{z}}^{k}} \right)$$

计算输出层误差
$${ {\mathbf{\delta }}^{L}}$$
：
$${ {\mathbf{\delta }}^{L}}={ {\nabla }_{\mathbf{a}}}C\odot { {\sigma }^{'}}\left( { {\mathbf{z}}^{L}} \right)$$

误差反传：对于k=L-1,L-2,…,2，计算
$${ {\mathbf{\delta }}^{k}}=\left( { {\left( { {\mathbf{w}}^{k+1}} \right)}^{T}}{ {\mathbf{\delta }}^{k+1}} \right)\odot { {\sigma }^{'}}\left( { {\mathbf{z}}^{k}} \right)$$

输出：w和b的梯度为
$$\frac{\partial C}{\partial w_{ij}^{k}}=\delta _{i}^{k}a_{j}^{k-1}$$
，
$$\frac{\partial C}{\partial b_{i}^{k}}\text{=}\delta _{i}^{k}$$

注：由于反向传播时，
$$\frac{\partial C}{\partial w_{ij}^{k}}$$
与
$$a_{j}^{k-1}$$
有关，因而前向计算时，需要保存每层每个神经元的输出
$$a_{j}^{k}$$
，方便反传时使用。
