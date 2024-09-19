# Creating An ONNX Model With An Initializer

## Introduction

This example creates an ONNX model containing a single Convolution node with weights.

**`Constant`s in ONNX GraphSurgeon are automatically exported as initializers in the ONNX graph.**

## Running the example

Generate the model and save it to `test_conv.onnx` by running:
```bash
python3 example.py
```

The generated model will look like this:

![../resources/02_test_conv.onnx.png](../resources/02_test_conv.onnx.png)

在这段代码中，`W`代表的是卷积层的权重（filters或kernels），而其第一个维度为5表示卷积层有5个过滤器（或称为卷积核）。在卷积神经网络（CNN）中，每个卷积核负责从输入数据中提取不同的特征。这些卷积核在输入数据上滑动，计算卷积核与输入数据的点积，从而生成输出特征图（feature maps）。

具体来说，`W`的形状为`(5, 3, 3, 3)`：
- 第一个维度为5，意味着有5个不同的卷积核，因此最终的输出特征图（output feature maps）将有5个通道，每个通道对应一个卷积核的输出。
- 第二个维度为3，表示每个卷积核将会在输入数据的3个通道上进行操作，这对应于输入`X`的通道数，也就是说每个卷积核都是针对RGB三通道的全彩色图片设计的。
- 最后两个维度`(3, 3)`指的是每个卷积核的高度和宽度，意味着这些卷积核在输入图片上以3x3的窗口进行滑动计算。

因此，在这个上下文中，`W`的第一个维度为5是为了定义输出特征图有5个通道，每个通道由输入数据与一个不同的卷积核卷积而来，从而能够捕获输入数据的不同特征。这是构建卷积神经网络时定义卷积层权重的典型方式。
