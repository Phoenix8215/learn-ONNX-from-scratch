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


def main():
    # 定义输入和输出的形状
    batch_size = 1
    seq_length = 128
    input_dim = 512
    num_heads = 8
    hidden_size = 512
    feedforward_size = 2048
    num_layers = 6

    input_shape = [batch_size, seq_length, input_dim]
    output_shape = [batch_size, seq_length, input_dim]

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

    # 创建 Transformer 层
    for layer in range(num_layers):
        layer_norm1_output = f"layer{layer}_norm1.output"
        attn_output = f"layer{layer}_attn.output"
        attn_output_add = f"layer{layer}_attn_add.output"
        layer_norm2_output = f"layer{layer}_norm2.output"
        ff_output = f"layer{layer}_ff.output"
        ff_output_add = f"layer{layer}_ff_add.output"

        # Layer Normalization 1
        ln1_scale = np.ones((input_dim,), dtype=np.float32)
        ln1_bias = np.zeros((input_dim,), dtype=np.float32)
        ln1_scale_name = f"layer{layer}_ln1_scale"
        ln1_bias_name = f"layer{layer}_ln1_bias"

        ln1_scale_initializer = create_initializer_tensor(
            name=ln1_scale_name,
            tensor_array=ln1_scale,
            data_type=onnx.TensorProto.FLOAT)
        ln1_bias_initializer = create_initializer_tensor(
            name=ln1_bias_name,
            tensor_array=ln1_bias,
            data_type=onnx.TensorProto.FLOAT)
        initializers.extend([ln1_scale_initializer, ln1_bias_initializer])

        ln1_node = onnx.helper.make_node(
            name=f"layer{layer}_ln1",
            op_type="LayerNormalization",
            inputs=[previous_output_name, ln1_scale_name, ln1_bias_name],
            outputs=[layer_norm1_output],
            epsilon=1e-5,
        )
        nodes.append(ln1_node)

        # Multi-Head Attention
        # 注意力机制的权重和偏置
        qkv_weight = np.random.randn(3 * input_dim, input_dim).astype(np.float32)
        qkv_bias = np.zeros((3 * input_dim,), dtype=np.float32)
        qkv_weight_name = f"layer{layer}_qkv_weight"
        qkv_bias_name = f"layer{layer}_qkv_bias"

        qkv_weight_initializer = create_initializer_tensor(
            name=qkv_weight_name,
            tensor_array=qkv_weight,
            data_type=onnx.TensorProto.FLOAT)
        qkv_bias_initializer = create_initializer_tensor(
            name=qkv_bias_name,
            tensor_array=qkv_bias,
            data_type=onnx.TensorProto.FLOAT)
        initializers.extend([qkv_weight_initializer, qkv_bias_initializer])

        # 输出投影的权重和偏置
        proj_weight = np.random.randn(input_dim, input_dim).astype(np.float32)
        proj_bias = np.zeros((input_dim,), dtype=np.float32)
        proj_weight_name = f"layer{layer}_proj_weight"
        proj_bias_name = f"layer{layer}_proj_bias"

        proj_weight_initializer = create_initializer_tensor(
            name=proj_weight_name,
            tensor_array=proj_weight,
            data_type=onnx.TensorProto.FLOAT)
        proj_bias_initializer = create_initializer_tensor(
            name=proj_bias_name,
            tensor_array=proj_bias,
            data_type=onnx.TensorProto.FLOAT)
        initializers.extend([proj_weight_initializer, proj_bias_initializer])

        # QKV 计算
        qkv_matmul_output = f"layer{layer}_qkv_matmul.output"
        qkv_matmul_node = onnx.helper.make_node(
            name=f"layer{layer}_qkv_matmul",
            op_type="MatMul",
            inputs=[layer_norm1_output, qkv_weight_name],
            outputs=[qkv_matmul_output],
        )
        nodes.append(qkv_matmul_node)

        qkv_bias_add_output = f"layer{layer}_qkv_bias_add.output"
        qkv_bias_add_node = onnx.helper.make_node(
            name=f"layer{layer}_qkv_bias_add",
            op_type="Add",
            inputs=[qkv_matmul_output, qkv_bias_name],
            outputs=[qkv_bias_add_output],
        )
        nodes.append(qkv_bias_add_node)

        # 分割 Q, K, V
        q_output = f"layer{layer}_q.output"
        k_output = f"layer{layer}_k.output"
        v_output = f"layer{layer}_v.output"

        # 创建 starts 和 ends 的初始值
        q_starts_name = f"layer{layer}_q_starts"
        q_ends_name = f"layer{layer}_q_ends"
        k_starts_name = f"layer{layer}_k_starts"
        k_ends_name = f"layer{layer}_k_ends"
        v_starts_name = f"layer{layer}_v_starts"
        v_ends_name = f"layer{layer}_v_ends"

        q_starts = np.array([0], dtype=np.int64)
        q_ends = np.array([input_dim], dtype=np.int64)
        k_starts = np.array([input_dim], dtype=np.int64)
        k_ends = np.array([2 * input_dim], dtype=np.int64)
        v_starts = np.array([2 * input_dim], dtype=np.int64)
        v_ends = np.array([3 * input_dim], dtype=np.int64)

        q_starts_initializer = create_initializer_tensor(
            name=q_starts_name,
            tensor_array=q_starts,
            data_type=onnx.TensorProto.INT64)
        q_ends_initializer = create_initializer_tensor(
            name=q_ends_name,
            tensor_array=q_ends,
            data_type=onnx.TensorProto.INT64)
        k_starts_initializer = create_initializer_tensor(
            name=k_starts_name,
            tensor_array=k_starts,
            data_type=onnx.TensorProto.INT64)
        k_ends_initializer = create_initializer_tensor(
            name=k_ends_name,
            tensor_array=k_ends,
            data_type=onnx.TensorProto.INT64)
        v_starts_initializer = create_initializer_tensor(
            name=v_starts_name,
            tensor_array=v_starts,
            data_type=onnx.TensorProto.INT64)
        v_ends_initializer = create_initializer_tensor(
            name=v_ends_name,
            tensor_array=v_ends,
            data_type=onnx.TensorProto.INT64)

        initializers.extend([
            q_starts_initializer, q_ends_initializer,
            k_starts_initializer, k_ends_initializer,
            v_starts_initializer, v_ends_initializer
        ])

        axes_name = f"layer{layer}_slice_axes"
        axes = np.array([-1], dtype=np.int64)
        axes_initializer = create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64)
        initializers.append(axes_initializer)

        # 创建 Slice 节点
        q_slice_node = onnx.helper.make_node(
            name=f"layer{layer}_q_slice",
            op_type="Slice",
            inputs=[qkv_bias_add_output, q_starts_name, q_ends_name, axes_name],
            outputs=[q_output],
        )
        nodes.append(q_slice_node)

        k_slice_node = onnx.helper.make_node(
            name=f"layer{layer}_k_slice",
            op_type="Slice",
            inputs=[qkv_bias_add_output, k_starts_name, k_ends_name, axes_name],
            outputs=[k_output],
        )
        nodes.append(k_slice_node)

        v_slice_node = onnx.helper.make_node(
            name=f"layer{layer}_v_slice",
            op_type="Slice",
            inputs=[qkv_bias_add_output, v_starts_name, v_ends_name, axes_name],
            outputs=[v_output],
        )
        nodes.append(v_slice_node)

        # 重塑 Q, K, V 以适应多头注意力
        def reshape_for_heads(name, input_name):
            reshape_output = f"{name}_reshape.output"
            transpose_output = f"{name}_transpose.output"
            reshape_node = onnx.helper.make_node(
                name=f"{name}_reshape",
                op_type="Reshape",
                inputs=[input_name, f"{name}_shape"],
                outputs=[reshape_output],
            )
            transpose_node = onnx.helper.make_node(
                name=f"{name}_transpose",
                op_type="Transpose",
                inputs=[reshape_output],
                outputs=[transpose_output],
                perm=[0, 2, 1, 3],
            )
            return [reshape_node, transpose_node], transpose_output

        head_dim = input_dim // num_heads
        q_shape = np.array([batch_size, seq_length, num_heads, head_dim], dtype=np.int64)
        k_shape = q_shape
        v_shape = q_shape

        q_shape_name = f"layer{layer}_q_shape"
        k_shape_name = f"layer{layer}_k_shape"
        v_shape_name = f"layer{layer}_v_shape"

        q_shape_initializer = create_initializer_tensor(
            name=q_shape_name,
            tensor_array=q_shape,
            data_type=onnx.TensorProto.INT64)
        k_shape_initializer = create_initializer_tensor(
            name=k_shape_name,
            tensor_array=k_shape,
            data_type=onnx.TensorProto.INT64)
        v_shape_initializer = create_initializer_tensor(
            name=v_shape_name,
            tensor_array=v_shape,
            data_type=onnx.TensorProto.INT64)
        initializers.extend([q_shape_initializer, k_shape_initializer, v_shape_initializer])

        q_nodes, q_reshaped = reshape_for_heads(f"layer{layer}_q", q_output)
        k_nodes, k_reshaped = reshape_for_heads(f"layer{layer}_k", k_output)
        v_nodes, v_reshaped = reshape_for_heads(f"layer{layer}_v", v_output)
        nodes.extend(q_nodes + k_nodes + v_nodes)

        # 计算注意力分数
        attn_matmul_qk_output = f"layer{layer}_attn_matmul_qk.output"
        attn_matmul_qk_node = onnx.helper.make_node(
            name=f"layer{layer}_attn_matmul_qk",
            op_type="MatMul",
            inputs=[q_reshaped, k_reshaped],
            outputs=[attn_matmul_qk_output],
        )
        nodes.append(attn_matmul_qk_node)

        # 缩放注意力分数
        scale_value = 1.0 / np.sqrt(head_dim)
        scale_name = f"layer{layer}_scale"
        scale_initializer = create_initializer_tensor(
            name=scale_name,
            tensor_array=np.array([scale_value], dtype=np.float32),
            data_type=onnx.TensorProto.FLOAT)
        initializers.append(scale_initializer)

        attn_scaled_output = f"layer{layer}_attn_scaled.output"
        attn_scale_node = onnx.helper.make_node(
            name=f"layer{layer}_attn_scale",
            op_type="Mul",
            inputs=[attn_matmul_qk_output, scale_name],
            outputs=[attn_scaled_output],
        )
        nodes.append(attn_scale_node)

        # 计算注意力权重（Softmax）
        attn_softmax_output = f"layer{layer}_attn_softmax.output"
        attn_softmax_node = onnx.helper.make_node(
            name=f"layer{layer}_attn_softmax",
            op_type="Softmax",
            inputs=[attn_scaled_output],
            outputs=[attn_softmax_output],
            axis=-1,
        )
        nodes.append(attn_softmax_node)

        # 计算注意力输出
        attn_matmul_v_output = f"layer{layer}_attn_matmul_v.output"
        attn_matmul_v_node = onnx.helper.make_node(
            name=f"layer{layer}_attn_matmul_v",
            op_type="MatMul",
            inputs=[attn_softmax_output, v_reshaped],
            outputs=[attn_matmul_v_output],
        )
        nodes.append(attn_matmul_v_node)

        # 恢复形状
        attn_transpose_output = f"layer{layer}_attn_transpose.output"
        attn_transpose_node = onnx.helper.make_node(
            name=f"layer{layer}_attn_transpose",
            op_type="Transpose",
            inputs=[attn_matmul_v_output],
            outputs=[attn_transpose_output],
            perm=[0, 2, 1, 3],
        )
        nodes.append(attn_transpose_node)

        attn_reshape_output = f"layer{layer}_attn_reshape.output"
        attn_reshape_shape = np.array([batch_size, seq_length, input_dim], dtype=np.int64)
        attn_reshape_shape_name = f"layer{layer}_attn_reshape_shape"
        attn_reshape_shape_initializer = create_initializer_tensor(
            name=attn_reshape_shape_name,
            tensor_array=attn_reshape_shape,
            data_type=onnx.TensorProto.INT64)
        initializers.append(attn_reshape_shape_initializer)

        attn_reshape_node = onnx.helper.make_node(
            name=f"layer{layer}_attn_reshape",
            op_type="Reshape",
            inputs=[attn_transpose_output, attn_reshape_shape_name],
            outputs=[attn_reshape_output],
        )
        nodes.append(attn_reshape_node)

        # 输出投影
        proj_matmul_output = f"layer{layer}_proj_matmul.output"
        proj_matmul_node = onnx.helper.make_node(
            name=f"layer{layer}_proj_matmul",
            op_type="MatMul",
            inputs=[attn_reshape_output, proj_weight_name],
            outputs=[proj_matmul_output],
        )
        nodes.append(proj_matmul_node)

        proj_bias_add_output = f"layer{layer}_proj_bias_add.output"
        proj_bias_add_node = onnx.helper.make_node(
            name=f"layer{layer}_proj_bias_add",
            op_type="Add",
            inputs=[proj_matmul_output, proj_bias_name],
            outputs=[proj_bias_add_output],
        )
        nodes.append(proj_bias_add_node)

        # 残差连接 1
        attn_output_add_node = onnx.helper.make_node(
            name=f"layer{layer}_attn_add",
            op_type="Add",
            inputs=[previous_output_name, proj_bias_add_output],
            outputs=[attn_output_add],
        )
        nodes.append(attn_output_add_node)

        # Layer Normalization 2
        ln2_scale = np.ones((input_dim,), dtype=np.float32)
        ln2_bias = np.zeros((input_dim,), dtype=np.float32)
        ln2_scale_name = f"layer{layer}_ln2_scale"
        ln2_bias_name = f"layer{layer}_ln2_bias"

        ln2_scale_initializer = create_initializer_tensor(
            name=ln2_scale_name,
            tensor_array=ln2_scale,
            data_type=onnx.TensorProto.FLOAT)
        ln2_bias_initializer = create_initializer_tensor(
            name=ln2_bias_name,
            tensor_array=ln2_bias,
            data_type=onnx.TensorProto.FLOAT)
        initializers.extend([ln2_scale_initializer, ln2_bias_initializer])

        ln2_node = onnx.helper.make_node(
            name=f"layer{layer}_ln2",
            op_type="LayerNormalization",
            inputs=[attn_output_add, ln2_scale_name, ln2_bias_name],
            outputs=[layer_norm2_output],
            epsilon=1e-5,
        )
        nodes.append(ln2_node)

        # Feedforward 第一层
        ff_weight1 = np.random.randn(feedforward_size, input_dim).astype(np.float32)
        ff_bias1 = np.zeros((feedforward_size,), dtype=np.float32)
        ff_weight1_name = f"layer{layer}_ff_weight1"
        ff_bias1_name = f"layer{layer}_ff_bias1"

        ff_weight1_initializer = create_initializer_tensor(
            name=ff_weight1_name,
            tensor_array=ff_weight1,
            data_type=onnx.TensorProto.FLOAT)
        ff_bias1_initializer = create_initializer_tensor(
            name=ff_bias1_name,
            tensor_array=ff_bias1,
            data_type=onnx.TensorProto.FLOAT)
        initializers.extend([ff_weight1_initializer, ff_bias1_initializer])

        ff_matmul1_output = f"layer{layer}_ff_matmul1.output"
        ff_matmul1_node = onnx.helper.make_node(
            name=f"layer{layer}_ff_matmul1",
            op_type="MatMul",
            inputs=[layer_norm2_output, ff_weight1_name],
            outputs=[ff_matmul1_output],
        )
        nodes.append(ff_matmul1_node)

        ff_bias1_add_output = f"layer{layer}_ff_bias1_add.output"
        ff_bias1_add_node = onnx.helper.make_node(
            name=f"layer{layer}_ff_bias1_add",
            op_type="Add",
            inputs=[ff_matmul1_output, ff_bias1_name],
            outputs=[ff_bias1_add_output],
        )
        nodes.append(ff_bias1_add_node)

        # 激活函数（ReLU）
        ff_relu_output = f"layer{layer}_ff_relu.output"
        ff_relu_node = onnx.helper.make_node(
            name=f"layer{layer}_ff_relu",
            op_type="Relu",
            inputs=[ff_bias1_add_output],
            outputs=[ff_relu_output],
        )
        nodes.append(ff_relu_node)

        # Feedforward 第二层
        ff_weight2 = np.random.randn(input_dim, feedforward_size).astype(np.float32)
        ff_bias2 = np.zeros((input_dim,), dtype=np.float32)
        ff_weight2_name = f"layer{layer}_ff_weight2"
        ff_bias2_name = f"layer{layer}_ff_bias2"

        ff_weight2_initializer = create_initializer_tensor(
            name=ff_weight2_name,
            tensor_array=ff_weight2,
            data_type=onnx.TensorProto.FLOAT)
        ff_bias2_initializer = create_initializer_tensor(
            name=ff_bias2_name,
            tensor_array=ff_bias2,
            data_type=onnx.TensorProto.FLOAT)
        initializers.extend([ff_weight2_initializer, ff_bias2_initializer])

        ff_matmul2_output = f"layer{layer}_ff_matmul2.output"
        ff_matmul2_node = onnx.helper.make_node(
            name=f"layer{layer}_ff_matmul2",
            op_type="MatMul",
            inputs=[ff_relu_output, ff_weight2_name],
            outputs=[ff_matmul2_output],
        )
        nodes.append(ff_matmul2_node)

        ff_bias2_add_output = f"layer{layer}_ff_bias2_add.output"
        ff_bias2_add_node = onnx.helper.make_node(
            name=f"layer{layer}_ff_bias2_add",
            op_type="Add",
            inputs=[ff_matmul2_output, ff_bias2_name],
            outputs=[ff_bias2_add_output],
        )
        nodes.append(ff_bias2_add_node)

        # 残差连接 2
        ff_output_add_node = onnx.helper.make_node(
            name=f"layer{layer}_ff_add",
            op_type="Add",
            inputs=[attn_output_add, ff_bias2_add_output],
            outputs=[ff_output_add],
        )
        nodes.append(ff_output_add_node)

        previous_output_name = ff_output_add

    # 输出层的 Layer Normalization
    ln_final_scale = np.ones((input_dim,), dtype=np.float32)
    ln_final_bias = np.zeros((input_dim,), dtype=np.float32)
    ln_final_scale_name = "ln_final_scale"
    ln_final_bias_name = "ln_final_bias"

    ln_final_scale_initializer = create_initializer_tensor(
        name=ln_final_scale_name,
        tensor_array=ln_final_scale,
        data_type=onnx.TensorProto.FLOAT)
    ln_final_bias_initializer = create_initializer_tensor(
        name=ln_final_bias_name,
        tensor_array=ln_final_bias,
        data_type=onnx.TensorProto.FLOAT)
    initializers.extend([ln_final_scale_initializer, ln_final_bias_initializer])

    ln_final_output = model_output_name
    ln_final_node = onnx.helper.make_node(
        name="ln_final",
        op_type="LayerNormalization",
        inputs=[previous_output_name, ln_final_scale_name, ln_final_bias_name],
        outputs=[ln_final_output],
        epsilon=1e-5,
    )
    nodes.append(ln_final_node)

    ########################## 创建 graph ##############################
    graph = onnx.helper.make_graph(
        name="transformer",
        inputs=[input],
        outputs=[output],
        nodes=nodes,
        initializer=initializers,
    )

    ########################## 创建 model ##############################
    model = onnx.helper.make_model(graph, producer_name="onnx-transformer-sample")
    model.opset_import[0].version = 17  # 使用 opset_version 17

    ########################## 验证 & 保存 model ##############################
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    print("成功创建 {}.onnx".format(graph.name))
    onnx.save(model, "transformer.onnx")

'''
    Input (batch_size=1, seq_length=128, input_dim=512)
    |
+-------------------------------------------+
|          Transformer Layer x6             |
|                                           |
|  LayerNormalization                       |
|    |                                      |
|  MultiHeadAttention                       |
|    |                                      |
|  Add (Residual Connection)                |
|    |                                      |
|  LayerNormalization                       |
|    |                                      |
|  FeedForward Network                      |
|    |                                      |
|  Add (Residual Connection)                |
+-------------------------------------------+
    |
LayerNormalization
    |
Output (batch_size=1, seq_length=128, input_dim=512)
'''

if __name__ == "__main__":
    main()
