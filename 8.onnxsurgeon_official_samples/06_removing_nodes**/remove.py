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
import onnx

graph = gs.import_onnx(onnx.load("model.onnx"))

fake_node = [node for node in graph.nodes if node.op == "FakeNodeToRemove"][0]

# 获取fake节点的输入节点
# Node 提供了 i() 和 o() 函数，参数可以是索引（默认是0）
# 这些函数提供了方便的方式，替代先获取输入/输出张量，再获取张量的输入/输出节点的传统方法。
# 例如，node.i() 等价于 node.inputs[0].inputs[0]
inp_node = fake_node.i()

# 将输入节点重新连接到fake节点的输出张量，这样示例图中的第一个Identity节点就可以跳过fake节点。
inp_node.outputs = fake_node.outputs
fake_node.outputs.clear()

# 将fake节点从图中完全移除
graph.cleanup()
onnx.save(gs.export_onnx(graph), "removed.onnx")

