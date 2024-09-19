import numpy as np
import onnx
from onnx import numpy_helper


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    initializer = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())
    return initializer


def conv_bn_relu(input_name, conv_weight_name, conv_bias_name, bn_scale_name, bn_bias_name,
                 bn_mean_name, bn_var_name, output_name, nodes, initializers,
                 conv_params, layer_name):
    """
    创建 Conv -> BatchNorm -> ReLU 的组合
    """
    conv_output = f"{layer_name}_conv_output"
    bn_output = f"{layer_name}_bn_output"
    relu_output = output_name

    # 卷积权重和偏置
    conv_weight = np.random.randn(*conv_params['weight_shape']).astype(np.float32)
    conv_bias = np.zeros((conv_params['weight_shape'][0],), dtype=np.float32)

    conv_weight_initializer = create_initializer_tensor(
        name=conv_weight_name,
        tensor_array=conv_weight,
        data_type=onnx.TensorProto.FLOAT)
    conv_bias_initializer = create_initializer_tensor(
        name=conv_bias_name,
        tensor_array=conv_bias,
        data_type=onnx.TensorProto.FLOAT)
    initializers.extend([conv_weight_initializer, conv_bias_initializer])

    # 卷积节点
    conv_node = onnx.helper.make_node(
        name=f"{layer_name}_conv",
        op_type="Conv",
        inputs=[input_name, conv_weight_name, conv_bias_name],
        outputs=[conv_output],
        kernel_shape=conv_params['kernel_shape'],
        pads=conv_params['pads'],
        strides=conv_params['strides']
    )
    nodes.append(conv_node)

    # BatchNorm 参数
    num_features = conv_params['weight_shape'][0]
    bn_scale = np.ones((num_features,), dtype=np.float32)
    bn_bias = np.zeros((num_features,), dtype=np.float32)
    bn_mean = np.zeros((num_features,), dtype=np.float32)
    bn_var = np.ones((num_features,), dtype=np.float32)

    bn_scale_initializer = create_initializer_tensor(
        name=bn_scale_name,
        tensor_array=bn_scale,
        data_type=onnx.TensorProto.FLOAT)
    bn_bias_initializer = create_initializer_tensor(
        name=bn_bias_name,
        tensor_array=bn_bias,
        data_type=onnx.TensorProto.FLOAT)
    bn_mean_initializer = create_initializer_tensor(
        name=bn_mean_name,
        tensor_array=bn_mean,
        data_type=onnx.TensorProto.FLOAT)
    bn_var_initializer = create_initializer_tensor(
        name=bn_var_name,
        tensor_array=bn_var,
        data_type=onnx.TensorProto.FLOAT)
    initializers.extend([
        bn_scale_initializer, bn_bias_initializer,
        bn_mean_initializer, bn_var_initializer
    ])

    # BatchNorm 节点
    bn_node = onnx.helper.make_node(
        name=f"{layer_name}_bn",
        op_type="BatchNormalization",
        inputs=[conv_output, bn_scale_name, bn_bias_name, bn_mean_name, bn_var_name],
        outputs=[bn_output],
        epsilon=1e-5,
        momentum=0.9
    )
    nodes.append(bn_node)

    # ReLU 节点
    relu_node = onnx.helper.make_node(
        name=f"{layer_name}_relu",
        op_type="Relu",
        inputs=[bn_output],
        outputs=[relu_output]
    )
    nodes.append(relu_node)

    return relu_output


def basic_block(input_name, output_name, nodes, initializers, in_channels, out_channels,
                stride, layer_name, downsample=False):
    """
    创建 ResNet 的基本残差块（BasicBlock）
    """
    # 第一个卷积层
    conv1_weight_name = f"{layer_name}_conv1_weight"
    conv1_bias_name = f"{layer_name}_conv1_bias"
    bn1_scale_name = f"{layer_name}_bn1_scale"
    bn1_bias_name = f"{layer_name}_bn1_bias"
    bn1_mean_name = f"{layer_name}_bn1_mean"
    bn1_var_name = f"{layer_name}_bn1_var"

    conv1_params = {
        'weight_shape': [out_channels, in_channels, 3, 3],
        'kernel_shape': [3, 3],
        'pads': [1, 1, 1, 1],
        'strides': [stride, stride]
    }

    relu1_output = f"{layer_name}_relu1_output"

    relu1_output = conv_bn_relu(
        input_name=input_name,
        conv_weight_name=conv1_weight_name,
        conv_bias_name=conv1_bias_name,
        bn_scale_name=bn1_scale_name,
        bn_bias_name=bn1_bias_name,
        bn_mean_name=bn1_mean_name,
        bn_var_name=bn1_var_name,
        output_name=relu1_output,
        nodes=nodes,
        initializers=initializers,
        conv_params=conv1_params,
        layer_name=f"{layer_name}_conv1"
    )

    # 第二个卷积层
    conv2_weight_name = f"{layer_name}_conv2_weight"
    conv2_bias_name = f"{layer_name}_conv2_bias"
    bn2_scale_name = f"{layer_name}_bn2_scale"
    bn2_bias_name = f"{layer_name}_bn2_bias"
    bn2_mean_name = f"{layer_name}_bn2_mean"
    bn2_var_name = f"{layer_name}_bn2_var"

    conv2_params = {
        'weight_shape': [out_channels, out_channels, 3, 3],
        'kernel_shape': [3, 3],
        'pads': [1, 1, 1, 1],
        'strides': [1, 1]
    }

    conv2_output = f"{layer_name}_conv2_output"

    conv2_output = conv_bn_relu(
        input_name=relu1_output,
        conv_weight_name=conv2_weight_name,
        conv_bias_name=conv2_bias_name,
        bn_scale_name=bn2_scale_name,
        bn_bias_name=bn2_bias_name,
        bn_mean_name=bn2_mean_name,
        bn_var_name=bn2_var_name,
        output_name=conv2_output,
        nodes=nodes,
        initializers=initializers,
        conv_params=conv2_params,
        layer_name=f"{layer_name}_conv2"
    )

    # 残差连接
    if downsample or in_channels != out_channels:
        # 下采样的 shortcut，需要调整输入的尺寸和通道数
        downsample_conv_weight_name = f"{layer_name}_downsample_conv_weight"
        downsample_conv_bias_name = f"{layer_name}_downsample_conv_bias"
        downsample_bn_scale_name = f"{layer_name}_downsample_bn_scale"
        downsample_bn_bias_name = f"{layer_name}_downsample_bn_bias"
        downsample_bn_mean_name = f"{layer_name}_downsample_bn_mean"
        downsample_bn_var_name = f"{layer_name}_downsample_bn_var"

        downsample_params = {
            'weight_shape': [out_channels, in_channels, 1, 1],
            'kernel_shape': [1, 1],
            'pads': [0, 0, 0, 0],
            'strides': [stride, stride]
        }

        downsample_output = f"{layer_name}_downsample_output"

        downsample_output = conv_bn_relu(
            input_name=input_name,
            conv_weight_name=downsample_conv_weight_name,
            conv_bias_name=downsample_conv_bias_name,
            bn_scale_name=downsample_bn_scale_name,
            bn_bias_name=downsample_bn_bias_name,
            bn_mean_name=downsample_bn_mean_name,
            bn_var_name=downsample_bn_var_name,
            output_name=downsample_output,
            nodes=nodes,
            initializers=initializers,
            conv_params=downsample_params,
            layer_name=f"{layer_name}_downsample"
        )
    else:
        downsample_output = input_name

    # Add 节点
    add_output = f"{layer_name}_add_output"
    add_node = onnx.helper.make_node(
        name=f"{layer_name}_add",
        op_type="Add",
        inputs=[conv2_output, downsample_output],
        outputs=[add_output]
    )
    nodes.append(add_node)

    # 最后的 ReLU
    relu_output = output_name
    relu_node = onnx.helper.make_node(
        name=f"{layer_name}_relu",
        op_type="Relu",
        inputs=[add_output],
        outputs=[relu_output]
    )
    nodes.append(relu_node)

    return relu_output


def main():
    # 定义输入和输出的形状
    batch_size = 1
    input_channels = 3
    input_height = 224
    input_width = 224
    num_classes = 1000

    input_shape = [batch_size, input_channels, input_height, input_width]
    output_shape = [batch_size, num_classes]

    ########################## 创建 input/output ################################
    model_input_name = "input0"
    model_output_name = "output0"

    input = onnx.helper.make_tensor_value_info(
        model_input_name,
        onnx.TensorProto.FLOAT,
        input_shape)

    output = onnx.helper.make_tensor_value_info(
        model_output_name,
        onnx.TensorProto.FLOAT,
        output_shape)

    nodes = []
    initializers = []

    previous_output_name = model_input_name

    # 初始卷积层
    conv1_weight_name = "conv1_weight"
    conv1_bias_name = "conv1_bias"
    bn1_scale_name = "bn1_scale"
    bn1_bias_name = "bn1_bias"
    bn1_mean_name = "bn1_mean"
    bn1_var_name = "bn1_var"

    conv1_params = {
        'weight_shape': [64, 3, 7, 7],
        'kernel_shape': [7, 7],
        'pads': [3, 3, 3, 3],
        'strides': [2, 2]
    }

    conv1_output = conv_bn_relu(
        input_name=previous_output_name,
        conv_weight_name=conv1_weight_name,
        conv_bias_name=conv1_bias_name,
        bn_scale_name=bn1_scale_name,
        bn_bias_name=bn1_bias_name,
        bn_mean_name=bn1_mean_name,
        bn_var_name=bn1_var_name,
        output_name="conv1_relu_output",
        nodes=nodes,
        initializers=initializers,
        conv_params=conv1_params,
        layer_name="conv1"
    )

    previous_output_name = conv1_output

    # MaxPool 层
    maxpool_output = "maxpool_output"
    maxpool_node = onnx.helper.make_node(
        name="maxpool",
        op_type="MaxPool",
        inputs=[previous_output_name],
        outputs=[maxpool_output],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1]
    )
    nodes.append(maxpool_node)

    previous_output_name = maxpool_output

    # 定义每个阶段的层数和通道数
    layers = [2, 2, 2, 2]  # ResNet-18 的层数配置
    channels = [64, 128, 256, 512]

    in_channels = 64

    # 构建 ResNet 的层
    for stage, num_blocks in enumerate(layers):
        out_channels = channels[stage]
        for block in range(num_blocks):
            stride = 1
            downsample = False
            if block == 0 and stage != 0:
                stride = 2
                downsample = True
            layer_name = f"layer{stage + 1}_{block + 1}"
            previous_output_name = basic_block(
                input_name=previous_output_name,
                output_name=f"{layer_name}_output",
                nodes=nodes,
                initializers=initializers,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                layer_name=layer_name,
                downsample=downsample
            )
            in_channels = out_channels

    # 全局平均池化层
    avgpool_output = "avgpool_output"
    avgpool_node = onnx.helper.make_node(
        name="avgpool",
        op_type="GlobalAveragePool",
        inputs=[previous_output_name],
        outputs=[avgpool_output]
    )
    nodes.append(avgpool_node)

    # Flatten 层
    flatten_output = "flatten_output"
    flatten_node = onnx.helper.make_node(
        name="flatten",
        op_type="Flatten",
        inputs=[avgpool_output],
        outputs=[flatten_output],
        axis=1
    )
    nodes.append(flatten_node)

    # 全连接层
    fc_weight_name = "fc_weight"
    fc_bias_name = "fc_bias"

    fc_weight = np.random.randn(num_classes, in_channels).astype(np.float32)
    fc_bias = np.zeros((num_classes,), dtype=np.float32)

    fc_weight_initializer = create_initializer_tensor(
        name=fc_weight_name,
        tensor_array=fc_weight,
        data_type=onnx.TensorProto.FLOAT)
    fc_bias_initializer = create_initializer_tensor(
        name=fc_bias_name,
        tensor_array=fc_bias,
        data_type=onnx.TensorProto.FLOAT)
    initializers.extend([fc_weight_initializer, fc_bias_initializer])

    fc_output = "fc_output"
    fc_node = onnx.helper.make_node(
        name="fc",
        op_type="Gemm",
        inputs=[flatten_output, fc_weight_name, fc_bias_name],
        outputs=[fc_output],
        alpha=1.0,
        beta=1.0,
        transB=1
    )
    nodes.append(fc_node)

    # 输出层
    output_node = onnx.helper.make_node(
        name="output",
        op_type="Identity",
        inputs=[fc_output],
        outputs=[model_output_name]
    )
    nodes.append(output_node)

    ########################## 创建 graph ##############################
    graph = onnx.helper.make_graph(
        name="resnet18",
        inputs=[input],
        outputs=[output],
        nodes=nodes,
        initializer=initializers,
    )

    ########################## 创建 model ##############################
    model = onnx.helper.make_model(graph, producer_name="onnx-resnet-sample")
    model.opset_import[0].version = 12  # 设置 opset 版本

    ########################## 验证 & 保存 model ##############################
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    print("成功创建 {}.onnx".format(graph.name))
    onnx.save(model, "resnet18.onnx")


if __name__ == "__main__":
    main()
