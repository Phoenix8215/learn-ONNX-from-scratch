#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx_graphsurgeon as gs
import numpy as np
import onnx

print("Graph.layer Help:\n{}".format(gs.Graph.layer.__doc__))

# 我们可以使用 `Graph.register()` 将一个函数添加到 Graph 类中。
# 之后我们可以直接在图实例上调用这个函数，比如 `graph.add(...)`
@gs.Graph.register()
def add(self, a, b):
    # Graph.layer 函数创建一个节点，添加输入和输出，并将其添加到图中。
    # 它返回节点的输出张量，以方便链式操作。
    # 该函数会在输入/输出字符串之前添加索引，以确保对 layer() 函数的多次调用生成不同的张量。
    # 但是，这并不保证与图中的其他张量不会发生重叠。因此，你应该选择合适的前缀，尽量避免冲突。
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])


@gs.Graph.register()
def mul(self, a, b):
    return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"])


@gs.Graph.register()
def gemm(self, a, b, trans_a=False, trans_b=False):
    attrs = {"transA": int(trans_a), "transB": int(trans_b)}
    return self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs)


# 你还可以在注册函数时指定一组 opsets。
# 默认情况下，函数会注册到所有低于 Graph.DEFAULT_OPSET 的 opset。
@gs.Graph.register(opsets=[11])
def relu(self, a):
    return self.layer(op="Relu", inputs=[a], outputs=["act_out_gs"])


# 注意，同一个函数可以针对不同的 opset 以不同的方式定义。
# 它只有在图的 opset 匹配到该函数注册的 opset 时才会被调用。
# 因此，对于这个例子中使用的 opset 11 的图，下面的函数永远不会被使用。
@gs.Graph.register(opsets=[12])
def relu(self, a):
    raise NotImplementedError("This function has not been implemented!")


##########################################################################################################
# 上面注册的函数极大地简化了图的构建过程。

graph = gs.Graph(opset=11)

# 生成一个计算如下公式的图：
# output = ReLU((A * X^T) + B) (.) C + D
X = gs.Variable(name="X", shape=(64, 64), dtype=np.float32)
graph.inputs = [X]

# axt = (A * X^T)
# 注意，我们可以直接使用 NumPy 数组（例如 Tensor A），
# 它们会自动转换为 Constants。
A = np.ones(shape=(64, 64), dtype=np.float32)
axt = graph.gemm(A, X, trans_b=True)

# dense = ReLU(axt + B)
B = np.ones((64, 64), dtype=np.float32) * 0.5
dense = graph.relu(*graph.add(*axt, B))

# output = dense (.) C + D
# 如果提供的是 Tensor 实例（例如 Tensor C），它不会被修改。
# 如果你希望设置图中张量的确切名称，应该手动构造张量，而不是传递字符串或 NumPy 数组。
C = gs.Constant(name="C", values=np.ones(shape=(64, 64), dtype=np.float32))
D = np.ones(shape=(64, 64), dtype=np.float32)
graph.outputs = graph.add(*graph.mul(*dense, C), D)

# 最后，我们需要设置输出的数据类型以确保这是一个有效的 ONNX 模型。
# 在我们的例子中，所有数据类型都是 float32。
for out in graph.outputs:
    out.dtype = np.float32

onnx.save(gs.export_onnx(graph), "model.onnx")

