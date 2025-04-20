---
title: 深度学习笔记(第二部分)
date: 2025-04-10 20:40:16
tags:
---

# 循环神经网络 (Recurrent Networks)

## 应用场景

之前 CNN 和 MLP 的输入和输出都只有一个,但实际问题中的输入和输出可能有多个。比如给图像加描述(one to many), 视频分类(many to one), 机器翻译(many to many),这些应用中可以使用循环神经网络。
![常见的任务种类](deeplearning-note-part2/tasktype.png)

当你处理的问题的输入**或**输出涉及到 _**序列**_ 时候，可以使用循环神经网络。

循环神经网络可以对**非序列型**的数据比如图片进行**序列化的处理**。

## 基础概念

我们可以通过在每一个时间步应用一个递推公式来处理一系列向量$x$。
其中使用相同的权重矩阵$W$和状态更新函数$f_W$就可以处理任意长的序列。

在序列被处理过程中，RNN 有一个一直更新的内部状态。

$$
h_{t}=f_{W} ( h_{t-1}, x_{t} )
$$

初始状态$h_0$被设置为全 0 或者通过学习得到。

![RNN](deeplearning-note-part2/RNN.png)

## RNN 实现

### Vanilla RNN

一种 RNN 是"Vanilla RNN"。

$$
h_{t}=t a n h ( W_{h h} h_{t-1}+W_{x h} x_{t}+B_{h} )
$$

每一步的输出$y_t$通过另一个权重矩阵和隐藏状态计算得出。

$$
y_{t}=W_{h y} h_{t}+B_{y}
$$

Vanilla RNN 在反向传播中有两个问题：

1. 经过了 tanh 函数处理，不容易反向传播。
2. 在对矩阵乘法反向传播时候要乘矩阵的转置，并且要乘很多次。如果矩阵最大的奇异值大于一，梯度会爆炸。如果矩阵最大的奇异值小于一，梯度会消失。

### 长短期记忆（Long Short-Term Memory，LSTM）

长短期记忆相较于 Vanilla RNN，每步有两个状态：**单元状态**和**隐藏状态**。
![长短期记忆与Vanilla RNN](deeplearning-note-part2/Vanillavslstm.png)

LSTM 内有四个门：

-   f 遗忘门：是否清除单元状态
-   i 输入门：是否写入单元
-   g 候选门（或更新门）：写入单元的内容有多少
-   o 输出门：从单元中输出多少信息

通过这四个门可以计算出**单元状态**和**隐藏状态**，**单元状态**是 LSTM 的内部状态，通过输出门来控制在**隐藏状态**中显示多少**单元状态**的信息。

![长短期记忆细节](deeplearning-note-part2/lstmdetail.png)

LSTM 的计算图顶部有一条方便梯度传播的“高速公路”，类似于 Resnet 的设计，便于进行优化。
![LSTM梯度](deeplearning-note-part2/lstmgradient.png)

### 其他实现

还有一种 RNN 网络经常被使用，称为**门控循环单元**(Gated Recurrent Unit,GRU)。人们通过神经网络也预测出了很多其他 RNN 网络，但 LSTM 和 GRU 的表现通常不错。

## 截断式时间反向传播（truncated backpropagation through time）

循环神经网络是在时间上传播。在整个序列上执行前向传播以计算损失，然后在整个序列上执行反向传播以计算梯度。

然而，当我们需要训练非常长的序列时，这种方法会变得非常棘手。在实际应用中，人们通常采用一种近似方法，称为**截断式时间反向传播**（truncated backpropagation through time）。

这种方法的做法是：只在序列的若干小片段上进行前向和反向传播，而不是在整个序列上进行。
![截断式时间反向传播](deeplearning-note-part2/truncatedbackprop.png)
当我们处理下一段的数据时，仍然会保留来自前一段的隐藏状态，并将其传递下去。前向传播过程不会受到影响。在每一段的反向传播结束后，对权重进行一次梯度更新

## 语言模型

语言模型用来预测下一个字符输出什么。模型在每一步从概率分布中采样得到输出，并作为下一步的输入。

在自然语言处理（NLP）中，经常需要将离散的符号（如字母、词语）转换为计算机能处理的数值形式，最简单的方法是使用独热编码（One-hot encoding）。每个向量只有一个位置是 1，其他位置都是 0。

这些向量维度高且稀疏，当矩阵相乘时，效率很低。为了解决这些问题，可以引入**嵌入层**。将每个离散的符号（如一个字母）映射到一个低维稠密向量中，并通过训练学习这个映射。

![RNN语言模型](deeplearning-note-part2/rnnlm.png)

## 多层 RNN

以上都是单层的 RNN，通过将一个 RNN 的隐藏状态作为输入传递给另一个 RNN，可以实现多层的 RNN。
![多层RNN](deeplearning-note-part2/mutilayerRNN.png)

## RNN 类型

### 一对多：

![one2many](deeplearning-note-part2/one2many.png)

### 多对一：

![many2one](deeplearning-note-part2/many2one.png)

### 多对多：

对于多对多的情况，我们在每一个时间步都计算一个$y_t$和对应的损失。最后，我们只需将所有时间步的损失相加，并将其作为整个网络的总损失。
![many2many](deeplearning-note-part2/many2many.png)

### 序列到序列

对于机器翻译这种序列到序列(seq2seq)、(many to many)的问题，可以将一个编码器（encoder, many to one）和一个解码器(decoder,one to many)的 RNN 接起来, 它们有各自单独的权重。

编码器:处理输入数据 $[x_1, x_2, x_3, ...]$。整个输入序列的信息压缩到**上下文向量**(Context Vector)中，通常设为最后一个隐藏状态$h_T$。

解码器:$h_T$ 作为解码器的初始隐藏向量。 每一步的输出作为下一步的输入。第一步输入通常直接指定为 \<start\>。当 decoder 输出采样到\<end\>表示结束。

![seq2seq](deeplearning-note-part2/seq2seq.png)

# 注意力机制

## RNN with attention

![序列到序列RNN模型问题](deeplearning-note-part2/seq2seqprob.png)
在之前[序列到序列](#序列对序列)的网络中，所有信息被压缩到**上下文向量**中，当输入比较长时候，这个向量不能够表示所有信息。可以将注意力机制用于这一模型,在每一步产生一个上下文向量。在每一步，解码器“注意”输入序列的不同部分。

计算过程如下：
![序列到序列RNN模型加注意力](deeplearning-note-part2/RNNwithattention.png)

1. 在每一步中，使用当前解码器状态$s_{t-1}$和每一个编码器的隐藏状态，计算出每个隐藏状态的对齐分数。可以用 MLP 计算。
2. 使用 softmax 得到概率分布，也就是注意力权重$a_{t,i}$。
3. 这一步的上下文变量 $c_{t}$ 是隐藏状态 $h_i$ 的线性组合。$\begin{array} {l} {\mathbf{c_{t}}=\Sigma_{\mathbf{i}} \mathbf{a_{t, i}} \mathbf{h_{i}}} \\ \end{array}$
4. 在编码器中使用这个上下文变量 $c_{t}$ 和输入计算出解码器下一个状态$s_{t}$

## CNN with attention

在利用 CNN 给图片加标注时候也可以使用注意力机制，每次看图片中不同的地方。
使用解码器当前状态和 CNN 得到的**特征图**也可以计算类似的对齐分数、注意力权重。
![图片标注加注意力机制](deeplearning-note-part2/CNNwithattention.png)

## 注意力层
