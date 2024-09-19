import torch
import torch.onnx
import onnxruntime

OperatorExportTypes = torch._C._onnx.OperatorExportTypes
# 我们按照正常的方式，创建一个图。不考虑自己设计算子的话，我们其实是直接导出这个onnx的
# 只不过onnx会为这个实现根据自己内部注册的各个算子，追踪每一个节点生成图
# 我们可以观察一下onnx，会比较复杂，我们管这个叫做算子的inline autograd function

class CustomOp(torch.autograd.Function):
    @staticmethod# 注意要加staticmethod表明其是一个静态方法
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        x = x.clamp(min=0)
        return x / (1 + torch.exp(-x)) # 用pytorch自定义一个算子,实现截取后进行sigmoid

customOp = CustomOp.apply

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = customOp(x)
        return x


def validate_onnx():
    input   = torch.rand(1, 50).uniform_(-1, 1).reshape(1, 2, 5, 5)

    # PyTorch的推理
    model = Model()
    x     = model(input)
    print("result from Pytorch is :\n", x)

    # onnxruntime的推理
    sess  = onnxruntime.InferenceSession('../models/sample-customOp.onnx')
    x     = sess.run(None, {'input0': input.numpy()})
    print("result from onnx is:    \n", x)

    from onnx.reference import ReferenceEvaluator
    import onnx
    onnxmodel = onnx.load('../models/sample-customOp.onnx')
    sess1 = ReferenceEvaluator(onnxmodel)
    feeds = {'input0': input.numpy()}
    print("result from onnx is:    ", sess1.run(None, feeds))

def export_norm_onnx():
    input   = torch.rand(1, 50).uniform_(-1, 1).reshape(1, 2, 5, 5)
    model   = Model()
    model.eval()

    # 我们可以在导出onnx的时候添operator_export_type的限制，防止导出的onnx进行inline
    file    = "../models/sample-customOp.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 12)
        # operator_export_type = OperatorExportTypes.ONNX_FALLTHROUGH)
    print("Finished normal onnx export")

if __name__ == "__main__":
    export_norm_onnx()

    # 自定义完onnx以后必须要进行一下验证
    validate_onnx()
