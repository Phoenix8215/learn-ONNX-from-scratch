# Learn ONNX from scratch
⚠️本项目是基于韩博的[tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)，增加了大量案例和注解
🥰代表一些知识点的简介，🚀代表实战案例，❗代表一些tips

## ONNX部分

1. [1.generate-onnx](1.generate-onnx)初步尝试：导出三个简单的onnx模型(两个输出头|动态shape) 

2. [2.export-onnx](2.export-onnx)使用onnx-simplifier对算子进行简化，观察算子融合现象(Conv+BN+Act)

3. [3.read-and-parse-onnx](3.read-and-parse-onnx)使用onnx提供的API一步步搭建onnx模型，解析onnx各个节点的信息(输入输出节点，shape，name…)，这部分主要的参考链接：[点我](https://onnx.ai/onnx/api/classes.html#)

4. [4.export-unsupported-node](4.export-unsupported-node)学习如何导出不被支持的算子或自定义算子，给出了以下4种情况下的处理方式：
   - onnx支持该算子，但是导出依旧失败，主要问题是 PyTorch 与 onnx 之间没有建立 asinh 的映射
   - 如何导出自定义的算子
   - 如何导出onnx中不被支持的算子(并未实现后端推理)
   - 动态指定SRCNN模型的上采样倍率，并导出onnx


🥰当出现导出onnx不成功的时候，我们需要考虑的事情，难易度从低到高：

- 修改opset的版本 
  -  查看不支持的算子在新的opset中是否被支持 
  - **注意：因为onnx是一种图结构表示，并不包含各个算子的实现。除非我们是要在onnx-runtime上测试， 否则我们更看重onnx-trt中这个算子的支持情况** 
- 替换pytorch中的算子组合 
  - 把某些计算替换成onnx可以识别的
- 在pytorch登记onnx中某些算子 
  - 有可能onnx中有支持，但没有被登记
- 直接修改onnx，创建plugin 
  - 使用onnx-surgeon 
  - 一般是用在加速某些算子上使用

5. [5.debugONNX](5.debugONNX)学习如何查看onnx中间层节点的输出

## `onnx-surgeon`部分

1. [8.onnxsurgeon_official_samples](8.onnxsurgeon_official_samples)根据`onnx-surgeon`官方代码，学习如何使用onnx-surgeon，代码比较容易理解，也对部分代码进行了解读

🥰`onnx-surgeon`更加方便的添加/修改`onnx`节点，更加方便的修改子图 ，更加方便的替换算子，底层一般是用的`onnx.helper`，但是给做了一些封装。

- onnx_graph_surgeon(gs)中的IR会有以下三种结构:

- Tensor有两种类型
  - Variable:  主要就是那些不到推理不知道的变量
  - Constant:  不用推理时，而在推理前就知道的变量

- Node:跟onnx中的NodeProto差不多

- Graph:跟onnx中的GraphProto差不多

🥰gs帮助我们隐藏了很多信息 :node的属性以前使用AttributeProto保存， 但是gs中统一用dict来保存。gs可以方便我们把整个网络中的一些子图给“挖”出来，以此来分析细节 ，一般配合`polygraphy`使用，去寻找量化掉精度严重的子图。 gs中最重要的一个特点，在于我们可以使用gs来替换算子或者创建算子 ，这个会直接跟后面的TensorRT plugin绑定，实现算子的加速或者不兼容算子的实现。

2. [7.onnxsurgeon](7.onnxsurgeon)使用onnx-surgeon替换LayerNormalization算子和min-max算子

3. [6.export-onnx-from-oss](6.export-onnx-from-oss)将swin transformer导出为onnx 

🥰导出了onnx不是重点，我们需要将这些onnx导出为TensorRT模型，并查看性能。我们有几种方式导出 ：

- trtexec命令行(快速测试推理引擎) 
- TensorRT python API (大量的单元测试) 
- TensorRT C++ API (结合其他的前处理后处理进行底层部署)

## 更多学习内容
- https://github.com/Phoenix8215/A-White-Paper-on-Neural-Network-Deployment
- https://zhuanlan.zhihu.com/p/651219124
- https://zhuanlan.zhihu.com/p/720745913


