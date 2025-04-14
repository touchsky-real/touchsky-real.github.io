---
title: 深度学习笔记(第二部分)
date: 2025-04-10 20:40:16
tags:
---

# 循环神经网络 (Recurrent Networks)

之前的 CNN 和 MLP 的输入和输出都是一,但输入和输出可能有多个。比如给图像加描述(one to many), 视频分类(many to one), 机器翻译(many to many),这些应用中可以使用循环神经网络。
![常见的任务种类](deeplearning-note-part2/tasktype.png)

当你处理的问题的输入**或**输出涉及到**_序列_**时候，可以使用循环神经网络。

循环神经网络可以对**非序列型**的数据比如图片进行**序列化的处理**。

## Vanilla RNN

"Vanilla RNN" 是指最基础的、没有任何改进的循环神经网络（RNN）结构，它使用相同的权重矩阵$W$。
![RNN](deeplearning-note-part2/RNN.png)

假设有一个输入序列为: $[x_1, x_2, x_3]$ , $W$为权重矩阵，$h$为隐藏状态。

计算时先更新隐藏状态，然后用更新后的隐藏状态计算输出：

1. 处理第一个输入信息: $h_1 = f(W_{xx} x_1 + W_{hh} h_0 + b)$
2. (Optional) 输出结果: $y_1 = W_{hy} h_1$
3. 处理第二个输入信息: $h_2 = f(W_{xx} x_2 + W_{hh} h_1 + b)$
4. (Optional) 输出结果: $y_2 = W_{hy} h_2$
5. 处理第三个输出信息: $h_3 = f(W_{xx} x_3 + W_{hh} h_2 + b)$
6. 输出结果: $y_3 = W_{hy} h_3$

> $f$ 是激活函数, $h_0$ 通常被初始化为全零向量

$y_1$和$y_2$是可选的是因为很多任务是 many to one 型的，只要最后的输出就行。

one to many RNN：
![one2many](deeplearning-note-part2/one2many.png)
many to one RNN：
![many2one](deeplearning-note-part2/many2one.png)
many to many RNN：
![many2many](deeplearning-note-part2/many2many.png)

对于机器翻译这种序列到序列(seq2seq)、(many to many)的问题，可以将一个编码器（encoder, many to one）和一个解码器(decoder,one to many)的 RNN 接起来, 它们有各自单独的权重。

编码器:

1. 处理输入数据 $[x_1, x_2, x_3, ...]$
2. 整个输入序列的信息压缩如最后一个隐藏状态 $h_T$ 中，该隐藏状态称为**上下文向量**(Context Vector)

解码器:

1. $h_T$ 作为解码器的初始隐藏向量
2. 每一步的输入数据为上一步的输出 $y_i$，由于 decoder 中的第一步没有上一个输出 y，所以通常直接指定为 \<start\>。当 decoder 输出采样到\<end\>表示结束。

![seq2seq](deeplearning-note-part2/seq2seq.png)
