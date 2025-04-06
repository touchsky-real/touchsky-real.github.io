---
title: 深度学习笔记
date: 2025-03-29 20:59:13
tags:
---

# 线性分类器（Linear Classifiers）

线性分类器的缺点
![线性分类器的缺点](deeplearning-note/Linear%20Classifiers%20Problem.png)

解决方法之一：**特征变换**
![特征变换](deeplearning-note/特征变换.png)

## 损失函数

损失函数定量描述模型的好坏，它代表了我们对于模型中参数的偏好。可能有两组参数的损失值相同，仅由数据计算而来的损失函数会认为两者相同。可以在损失函数中加入正则项来体现出人类的先验的对参数的偏好。

# 优化（Optimization）

$$
w^{*}=\operatorname{a r g} \operatorname* {m i n}_{w} L ( w )
$$

## SGD

对 gradient descent 进行 Stochastic 处理，每次迭代时候抽取一批样本而不是用全部样本用于参数更新来降低算力要求。

$$
x_{t+1}=x_{t}-\alpha\nabla f ( x_{t} )
$$

```python
for t in range(num_steps):
    dw = compute_gradient(w)
    w -= learning_rate * dw
```

### 问题

![problem of SGD](deeplearning-note/SGDproblem.png)

## SGD with Momentum

SGD with Momentum 是为了克服 SGD 在收敛的过程中可能会停在 **局部最小值** 或者 **鞍点** 的问题，在这些点处梯度为 0，参数无法继续更新。

通过给 SGD 一个速度，从而越过**局部最小值** 或者 **鞍点** 可以解决这些问题。

![SGD with Momentum](deeplearning-note/SGDwithMomentum.png)

$$
\begin{aligned}
v_{t+1} &= \rho v_{t} + \nabla f ( x_{t} ) \\
x_{t+1} &= x_{t} - \alpha v_{t+1}
\end{aligned}
$$

```python
v = 0
for t in range(num_steps):
    dw = compute_gradient(w)
    v = rho * v + dw
    w -= learning_rate * v
```

Build up "velocity" as a running mean of gradients. Rho gives"friction";typically rho=0.9 or 0.99

等价于

$$
\begin{aligned}
  v_{t+1} &= \rho v_{t}-\alpha\nabla f ( x_{t} ) \\
  x_{t+1} &= x_{t}+v_{t+1}
\end{aligned}
$$

```python
v = 0  # 初始化动量项
for t in range(num_steps):  # 迭代 num_steps 次
    dw = compute_gradient(w)  # 计算当前权重 w 的梯度
    v = rho * v - learning_rate * dw  # 计算新的动量值
    w += v  # 更新权重
```

## Nesterov Momentum

根据速度向量到达新的点后计算梯度，对这个梯度和原来的速度进行向量和，作为原来的点更新使用的梯度。
![Nesterov Momentum graph](deeplearning-note/NesterovMomentum.png)

$$
\begin{array} {l} {v_{t+1}=\rho v_{t}-\alpha\nabla f ( x_{t}+\rho v_{t} )} \\ {x_{t+1}=x_{t}+v_{t+1}} \\ \end{array}
$$

```python
V = 0  # 初始化动量
for t in range(num_steps):
    dw = compute_gradient(w)  # 计算梯度
    old_v = V  # 记录旧的动量
    V = rho * V - learning_rate * dw  # 更新动量
    w -= rho * old_v - (1 + rho) * V  # 更新权重
```

## AdaGrad 算法（Adaptive Gradient Algorithm）

沿着“陡峭”方向的进展受到抑制，而沿着“平坦”方向的进展被加速。

```python
grad_squared = 0  # 初始化梯度累积项
for t in range(num_steps):
    dw = compute_gradient(w)  # 计算梯度
    grad_squared += dw * dw  # 累积梯度平方
    w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)  # 进行参数更新，避免除零

```

问题：grad_squared 可能会在到达损失函数最低点过大，而使得参数停止更新。
解决方法：**RMSProp**。

## RMSProp

与 AdaGrad 算法相比增加了一个“摩擦”项。

```python
grad_squared = 0  # 初始化累积梯度平方项
for t in range(num_steps):
    dw = compute_gradient(w)  # 计算梯度
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dw * dw  # 计算加权移动平均
    w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)  # 更新参数

```

## Adam 算法

结合两个好的 idea: SGD with Momentum + RMSProp = Adam
但是刚开始的时候梯度可能过大。
优化：**偏差修正**。

```python
moment1 = 0  # 一阶矩估计（动量项）
moment2 = 0  # 二阶矩估计（梯度平方的指数加权平均）

for t in range(num_steps):
    dw = compute_gradient(w)  # 计算梯度

    # 计算一阶矩估计（动量）
    moment1 = beta1 * moment1 + (1 - beta1) * dw

    # 计算二阶矩估计（梯度平方的移动平均）
    moment2 = beta2 * moment2 + (1 - beta2) * dw * dw

    # 计算一阶和二阶矩的偏差修正
    moment1_unbias = moment1 / (1 - beta1 ** t)
    moment2_unbias = moment2 / (1 - beta2 ** t)

    # 更新参数
    w -= learning_rate * moment1_unbias / (moment2_unbias.sqrt() + 1e-7)

```

### 经验

> Adam with beta1 = 0.9
> beta2 = 0.999, and learning rate = 1e-3, 5e-4, 1e-4 is a great starting point for many models!

# 神经网络

## 特征变换

通过对原始的输入特征进行变换，可能就能够处理原来的模型所不能处理的问题。对于图像分类而言，常用的特征变换包括：

-   Color Histogram 色彩直方图
-   Histogram of Oriented Gradients (HoG) 方向梯度直方图 (HoG)
-   Bag of Words 词袋模型 **_数据驱动的_**

不同的特征变换可以组合在一起使用。

![image Features vs Neural Networks](deeplearning-note/image%20Features%20vs%20Neural%20Networks.png)

## 端到端的学习

先进行特征提取再利用模型在提取出的特征上对图像进行分类时候，只会调整模型的参数，而特征提取的部分可能不会提高图片的分类效果。
因此，更好的办法是端到端的学习，输入原始数据，输出想要的结果。中间整体地对特征提取和特征处理部分进行训练来提高图像分类效果。

## 基础概念

深度神经网络的层数通常是指网络所含权重矩阵的个数。
宽度是隐藏表示的纬度。隐藏表示的纬度通常是一样的。
激活函数可以看成两个权重矩阵之间的“三明治”，给予网络额外的表现能力。

激活函数有多种。
![激活函数的种类](deeplearning-note/Activation%20Functions.png)

通常应该使用某种可调正则化参数的神经网络模型，而不是直接依赖网络本身的大小作为正则化因子。
网络大小不一定是最优正则化方式：虽然较大的网络可能更容易过拟合，但仅仅减少参数数量并不总是最佳策略。

## 反向传播

![反向传播](deeplearning-note/Backpropagation.png)
pytorch 中模型的计算步骤存储在**计算图**中，每个节点代表一次运算。
反向传播中，对于计算图中每个节点来说 downstream gradient = upstream gradient \* local gradient
在代码中：正向传播与反向传播的代码通常一一对应，但是顺序相反。

# 卷积网络

卷积网络中的权重矩阵一般称为 **卷积核** 或者 **filter**，它的深度一般和输入张量的深度一致，比如说 3。
输入张量和卷积核卷积后的结果被称为 **activation map**。
一层卷积层可以有多个卷积核，这是一个 可以设置的超参数。
![卷积操作](deeplearning-note/Convolutional%20filter.png)

有两种方式看待卷积后的结果：

1. 一系列的 feature map 的集合。
2. 特征向量组成的网格。

通常对一批图像进行处理。
![一般的卷积操作](deeplearning-note/generlized%20Convolution%20computation.png)

> 一般可以对第一层的参数可视化进行解释。

## 步幅和填充

padding: 在图片周围填充来防止图片尺寸缩小
特例：**same padding**后图像的大小不会改变
![same padding](deeplearning-note/same%20padding.png)

stride:下采样，防止网络需要很多层卷积才能获取到输入图片的全局信息。
除了 conv 中的 stride 可以下采样，池化层也可以下采样。

卷积通常的参数设置：
![卷积通常的参数设置](deeplearning-note/convolution%20common%20setting.png)

## 全连接层和 1x1 卷积的区别

全连接层可以用来破坏空间结构，比如网络最后一层生成分数。
1x1 卷积用来调节通道深度。

## 归一化

问题：网络很难训练。解决方法：归一化
通常使用 **批量归一化** ，使得每一层的输出符合均值为 0，方差为 1 的分布。
批量归一化训练和推理时行为不一致。
训练时：
![批量归一化](deeplearning-note/batch%20norm%20in%20train.png)
推理时：
![批量归一化](deeplearning-note/batch%20norm%20in%20test.png)

**批量归一化** 中一个批次的样本之间相互影响，**层归一化**可以避免这一问题。

## VGG net

两个 3x3 的卷积比单个 5x5 的卷积在参数、浮点计算更低的情况下效果可能会更好。
用卷积 stage 替换卷积层。每个 stage 里有多个卷积。
通过减半空间大小和把通道数翻倍，保持每个卷积 stage 中浮点计算次数差不多。

> 下采样：任何能够减少输入的空间尺寸的操作

Tips: 在实际应用中，不应该自己设计新的网络架构，而是应该在现有好的网络基础上修改。
![网络选择](deeplearning-note/architecture%20choice.png)

# 框架

## 静态计算图与动态计算图

静态计算图构建好后不会改变，动态计算图在每次前向传播中会构建新的计算图。

### 区别

优化区别
![静态动态优化区别](deeplearning-note/staticvsdynamicop.png)
序列化区别
![静态动态序列化区别](deeplearning-note/staticvsdynamicse.png)
调试区别
![静态动态调试区别](deeplearning-note/staticvsdynamicde.png)

## Pytorch

Pytoch 有三个抽象层次：

-   张量
-   自动微分
-   模块

![pytorch抽象层次](deeplearning-note/pytorchabstractionlevel.png)

代码实例：
![pytorch代码实例](deeplearning-note/pytorchcodeexample.png)

`with torch.no_grad():`告诉 pytoch 不要为上下文管理器中的操作构建计算图，通常**梯度更新**和**置零**不需要反向传播来计算梯度。

通过继承 nn.module 可以很方便地自定义网络。
![自定义网络](deeplearning-note/customizemodule.png)

pytorch 可以很方便地下载并使用预训练好的模型，通常 resnet 效果不错。
![预训练好的模型](deeplearning-note/pretrainedmodels.png)

pytorch 默认使用动态计算图。动态计算图使你可以在前向传播中使用控制语句，比如根据 loss 的不同选择为一个线性层选择不同的权重矩阵。
![动态图优点](deeplearning-note/dynamicgraphpro.png)

pytorch 可以使用静态图（也可以使用装饰器装饰 model 函数）。
![pytorch静态图](deeplearning-note/staticgraphinpytorch.png)

Pytoch 的张量操作中有任何一个输入张量的`require_grads`属性为`True`，pytorch 会为这个操作构建一部分计算图，并且操作的输出张量中`require_grads`属性也被 pytorch 设置为`True`。

## TensorFlow

`TensorFlow1.0`主要用静态计算图，`TensorFlow2.0`主要用动态计算图。

TensorFlow 中的`keras`类似于 pytoch 中的 nn 模块，提供模块级别的抽象。

TensorFlow 中的`tensorboard`很好用，是一个用来追踪网络统计信息的`web server`，pytorch 在`torch.utils.tensorboard`也提供了对 tensorboard 的支持。

# 网络的训练

## 激活函数

![激活函数的种类](deeplearning-note/Activation%20Functions.png)

### Sigmoid

问题:

1. 饱和的神经元杀死梯度。当输入的 x 值过大和过小时候，Sigmoid 函数的 local gradient 非常小。
2. 函数输出不是以零为中心。优化时候会走弯路，采用 minibatch 可以缓解这一点。
3. 指数运算成本高。

### Dead ReLU

当某个神经元（或者说某一层的某个输出通道）在训练过程中，**_无论_** 输入什么数据，它的输出永远是 0，那这个神经元就叫“死亡”了。

原因是：这个神经元的输入总是小于等于 0。

在 ReLU 的负区间，梯度是 0，反向传播时无法更新权重，那么训练过程中，这个神经元“永远不会再激活”。

采用 Leaky ReLU 可以避免这一问题，但是 Leaky ReLU 中有超参数，可以使网络自动学习这个参数。

### 总结

![激活函数总结](deeplearning-note/activationfuncsummary.png)
**不要**使用 Sigmoid 或者 tanh,现代的激活函数效果都差不多。

## 网络中的张量

常见的层包括:

1. 全连接层（Linear / Dense）

    - 输入：一个向量，比如大小为 (batch_size, input_dim)
    - 层定义：nn.Linear(input_dim, output_dim)
    - 输出：一个向量，形状为 (batch_size, output_dim)

        > `nn.Linear(input_dim, output_dim)`中神经元数量就等于`output_dim`。每个神经元参数数为`input_dim + 1`（权重 + 偏置）。每个神经元都接收所有 `input_dim `个输入特征。

2. 卷积层（Conv2d）

    - 输入：图像/特征图，形状为 (batch_size, in_channels, H, W)
    - 层定义：nn.Conv2d(in_channels, out_channels, kernel_size)
    - 输出：形状为 (batch_size, out_channels, H_out, W_out)

3. 激活层（ReLU、Sigmoid、Tanh 等）

    - 不会改变维度，只是逐元素地变换张量的值。

4. Flatten 层

    - 把 tensor 变成一个向量，用于送进全连接层。

## 数据预处理

图像中的像素值(0-255)一般都是正数，梯度就都是正的或是负的，不利于梯度更新。

对于 **图像** ，可以把数据集移到中心，调整方差。
![图像的数据预处理](deeplearning-note/imagepreprocess.png)

对于 **非图像** ，可以旋转数据集，使得特征之间不相互关联。
![非图像的数据预处理](deeplearning-note/nonimagepreprocess.png)

所有必须保持一致的转换（如标准化、编码）在训练和测试都做，但用**训练集**的统计量。因为在真实的世界中，没有训练集给你，没办法计算统计量。

不同色彩空间的图片之间的转换关系简单，网络可以很容易学习到，通常用 RGB 空间即可。

常用于图像的数据预处理:
![常用于图像的数据预处理](deeplearning-note/datapreprocessingimg.png)

## 参数初始化

> 在神经网络中，**激活值** 就是神经元的输出(经过激活函数处理后)，反映的是每个神经元“有没有响应”、“响应强不强”。

对于 ReLU 函数来说，如果激活值是 0，那么梯度就是 0。但是对于 tanh 和 sigmoid 来说不一定。

初始化的目的是 **为了让梯度良好便于优化** 。

用常数去初始化参数会使得梯度非常差，而用高斯分布去初始化网络仅适用于浅层网络，不适用于深层网络。可以采用 Xavier 和 MSRA 初始化。

### Xavier 初始化

对于 tanh 激活函数可以用 Xavier 初始化。原理是保持前后层方差一致。

对于全连接层。高斯分布的方差 std = 1/sqrt(Din)。对于卷积层来说 Din = 卷积核大小的平方\*输入通道数

### MSRA 初始化

对于 Relu 激活函数要进行修正，乘以 2。
高斯分布的方差 std = 2 /sqrt(Din)

对于残差网络的参数来说:
![残差网络参数初始化](deeplearning-note/residualinit.png)

## 正则化

### Dropout

通常对于线性层会使用 Dropout 进行正则化。

Dropout 会以概率 p（通常是 0.5），在每次前向传播中，随机将某一层的每个神经元的激活值置为 0，即“丢弃”该神经元，使其在当前这一轮不参与计算。这避免了特征之间的过度依赖。另一种观点是 Dropout 是训练了一大堆权重相同的小模型，每个小模型是完整模型的一部分。

在测试时候，必须缩放激活值使得：测试时候的输出等于训练时候输出的均值。有一种逆 Dropout 是在训练时候缩放来减少测试时候对设备的算力要求。

正则化通常在训练时加入某种随机性，在测试时候平均掉这种随机性，比如 batch normalization。
![正则化共性](deeplearning-note/regularizationcommonpatern.png)

### 数据增强

对于图片来讲，通过随机剪切、翻转等操作可以引入随机性，但通常不被认为是一种正则化。数据增强种类非常多，可以根据问题引入合适的操作。数据增强可以扩大数据集。

### 总结

Resnet 以后，现在通常只用 l2 正则、batch normalization 和数据增强。
![正则化总结](deeplearning-note/regularizationsummary.png)
