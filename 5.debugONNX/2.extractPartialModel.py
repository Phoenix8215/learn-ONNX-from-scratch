import onnx
# 把原计算图从边 /convs1/convs1.1/Conv_output_0 到边 /Add_output_0 的子图提取出来，并组成一个子模型。
# onnx.utils.extract_model
#  就是完成子模型提取的函数，它的参数分别是原模型路径、输出模型路径、子模型的输入边（输入张量）、子模型的输出边（输出张量）。
onnx.utils.extract_model('whole_model.onnx', 'partial_model.onnx', ['/convs1/convs1.1/Conv_output_0'], ['/Add_output_0'])