import onnx_graphsurgeon as gs
import numpy as np
import onnx


from onnx import shape_inference
model = shape_inference.infer_shapes(onnx.load("whole_model.onnx"))
#
# 重要提示：在某些情况下，ONNX形状推断可能无法正确推断形状，
# 这将导致子图无效。为了避免这种情况，你可以手动修改张量，自己添加形状信息。

# model = onnx.load("whole_model.onnx")
graph = gs.import_onnx(model)

# 由于我们已经知道了我们感兴趣的张量名称，我们可以
# 直接从张量映射中获取它们。
#
# 注意：如果你不知道想要的张量名称，你可以使用Netron查看图，
# 或者在交互式shell中使用ONNX GraphSurgeon打印图信息。
tensors = graph.tensors()
# 如果你想嵌入形状信息，但不能使用ONNX形状推断，
# 你可以在这个时候手动修改张量：
#
# graph.inputs = [tensors["x1"].to_variable(dtype=np.float32, shape=(1, 3, 224, 224))]
# graph.outputs = [tensors["add_out"].to_variable(dtype=np.float32, shape=(1, 3, 224, 224))]
#
# 重要提示：如果输入和输出张量中没有类型信息，必须为其添加类型信息。
#
# 注意：ONNX GraphSurgeon也接受动态形状 - 只需将相应的维度设置为`gs.Tensor.DYNAMIC`，
# 例如 `shape=(gs.Tensor.DYNAMIC, 3, 224, 224)`
graph.inputs = [tensors['/convs1/convs1.1/Conv_output_0'].to_variable(dtype=np.float32)]
graph.outputs = [tensors['/Add_output_0'].to_variable(dtype=np.float32)]

# 注意，我们不需要手动修改图中的其他部分。ONNX GraphSurgeon会自动处理删除
# 任何不必要的节点或张量，最终只剩下子图。
graph.cleanup()

onnx.save(gs.export_onnx(graph), "subgraph.onnx")
