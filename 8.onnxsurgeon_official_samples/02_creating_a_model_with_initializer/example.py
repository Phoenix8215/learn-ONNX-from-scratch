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

X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 224, 224))
# Since W is a Constant, it will automatically be exported as an initializer(in onnx)
W = gs.Constant(name="W", values=np.ones(shape=(5, 3, 3, 3), dtype=np.float32))

Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 5, 222, 222))

'''
output shape = (224 - 3 + 2 * 0) / 1 + 1
'''
node = gs.Node(op="Conv", inputs=[X, W], outputs=[Y])

# Note that initializers do not necessarily have to be graph inputs
graph = gs.Graph(nodes=[node], inputs=[X], outputs=[Y])
onnx.save(gs.export_onnx(graph), "test_conv.onnx")
