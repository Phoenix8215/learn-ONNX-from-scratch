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


# 在这里我们注册一个函数来处理所有子图替换的工作。
# 注意：由于注册的函数是完全可重用的，因此将它们重构到一个独立的模块中，
# 这样你可以在所有模型中使用它们。
@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    # 断开所有输入张量的输出节点
    # inputs 是 Tensor 类型
    for inp in inputs:
        inp.outputs.clear()

    # 断开所有输出张量的输入节点
    for out in outputs:
        out.inputs.clear()

    # 插入新节点。
    return self.layer(op="Clip", inputs=inputs, outputs=outputs)


# 现在我们进行实际的替换
graph = gs.import_onnx(onnx.load("model.onnx"))

tmap = graph.tensors()
# 你可以通过 Netron 确定输入和输出张量。在我们的例子中：
# 输入: [inp, MIN_VAL, MAX_VAL]
# 输出: [max_out]
inputs = [tmap["identity_out_0"], tmap["onnx_graphsurgeon_constant_5"], tmap["onnx_graphsurgeon_constant_2"]]
# inputs = [tmap["identity_out_0"], tmap["onnx_graphsurgeon_constant_2"]]
outputs = [tmap["max_out_6"]]

graph.replace_with_clip(inputs, outputs)

# 移除现在已经悬空的子图。
graph.cleanup().toposort()

# 完成！
onnx.save(gs.export_onnx(graph), "replaced.onnx")

