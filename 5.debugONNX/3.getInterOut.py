import onnx
import onnxruntime
import numpy as np
from onnx import helper, TensorProto

model = onnx.load('whole_model.onnx')

intermediate_tensor_names = ['/convs1/convs1.1/Conv_output_0']

# 为每个中间层输出添加新的输出节点
for tensor_name in intermediate_tensor_names:
    intermediate_value_info = helper.make_tensor_value_info(
        tensor_name,
        TensorProto.FLOAT,
        None  # 如果知道张量形状，可以填写具体形状
    )
    model.graph.output.append(intermediate_value_info)


onnx.save(model, 'model_with_intermediate_outputs.onnx')


session = onnxruntime.InferenceSession('model_with_intermediate_outputs.onnx')

input_name = session.get_inputs()[0].name

input_data = np.random.randn(1, 3, 20, 20).astype(np.float32)

output_names = [output.name for output in session.get_outputs()]

outputs = session.run(output_names, {input_name: input_data})

for name, output in zip(output_names, outputs):
    if name == "31":
        continue
    print(f'Output {name}: shape {output}')
