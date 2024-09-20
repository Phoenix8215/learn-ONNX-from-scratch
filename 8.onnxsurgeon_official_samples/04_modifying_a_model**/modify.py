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

graph = gs.import_onnx(onnx.load("model.onnx"))

# 1. 移除加法节点（第一个"Add"节点）的 `b` 输入
first_add = [node for node in graph.nodes if node.op == "Add"][0]
first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"]

# 2. 将加法操作改为 LeakyRelu
first_add.op = "LeakyRelu"
first_add.attrs["alpha"] = 0.02

# 3. 在加法节点后添加一个Identity节点
identity_out = gs.Variable("identity_out", dtype=np.float32)
identity = gs.Node(op="Identity", inputs=first_add.outputs, outputs=[identity_out])
graph.nodes.append(identity)

# 4. 将图的输出修改为 Identity 节点的输出
graph.outputs = [identity_out]
# 实做的时候发现input:x2并没有删去,要加下面这段代码
fisrtMul = [node for node in graph.nodes if node.op == "Mul"][0]
graph.inputs = [inp for inp in fisrtMul.inputs if inp.name != "a"]

# 5. 移除未使用的节点/张量，并对图进行拓扑排序
# ONNX 要求节点按拓扑顺序排列才能被认为是有效的。
# 因此，当你以乱序方式添加新节点时，通常需要对图进行排序。
# 在这种情况下，Identity节点已经在正确的位置（它是最后一个节点，并被追加到列表末尾），
# 但为了安全起见，我们仍然可以进行排序。
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "modified.onnx")
