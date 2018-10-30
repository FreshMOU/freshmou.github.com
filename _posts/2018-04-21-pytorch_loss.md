---
layout: post
title: Pytorch自定义Loss函数
date: 2018-04-21
tags: pytorch
---

Pytorch定义自己的loss函数十分方便，有很多方法可以选择。

### 只定义loss函数的前向计算公式

在pytorch中定义了前向计算的公式，在训练时它会自动帮你计算反向传播。

```python
import torch.nn as nn
Class YourLoss(nn.Module):
    def __init__():
        pass

    def forward():
        pass
```

### 自定义loss函数的forward和backward

```python
from numpy.fft import rfft2, irfft2

class BadFFTFunction(Function):

    def forward(self, input):
        numpy_input = input.numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)
```

### 自己写一个pytorch的C扩展

这个了解不多，所以也不太会

### 简单定义

看网上有说直接定义一个简单函数就可以了，可以尝试一下，与只定义forward类似。

```python
import torch
...... #模型操作
loss = torch.sum(x - y)
```