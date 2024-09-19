import onnx


def main(): 

    # model = onnx.load("../models/sample-convnet.onnx")
    model = onnx.load("sample-convnet.onnx")
    onnx.checker.check_model(model)

    graph        = model.graph
    initializers = graph.initializer
    nodes        = graph.node
    inputs       = graph.input
    outputs      = graph.output

    print("\n**************parse input/output*****************")
    for input in inputs:
        input_shape = []
        for d in input.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                input_shape.append(None)
            else:
                input_shape.append(d.dim_value)
        print("Input info: \
                \n\tname:      {} \
                \n\tdata Type: {} \
                \n\tshape:     {}".format(input.name, input.type.tensor_type.elem_type, input_shape))

    for output in outputs:
        output_shape = []
        for d in output.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                output_shape.append(None)
            else:
                output_shape.append(d.dim_value)
        print("Output info: \
                \n\tname:      {} \
                \n\tdata Type: {} \
                \n\tshape:     {}".format(input.name, output.type.tensor_type.elem_type, input_shape))

    print("\n**************parse node************************")
    for node in nodes:
        print("node info: \
                \n\tname:      {} \
                \n\top_type:   {} \
                \n\tinputs:    {} \
                \n\toutputs:   {}".format(node.name, node.op_type, node.input, node.output))

    print("\n**************parse initializer*****************")
    for initializer in initializers:
        print("initializer info: \
                \n\tname:      {} \
                \n\tdata_type: {} \
                \n\tshape:     {}".format(initializer.name, initializer.data_type, initializer.dims))


if __name__ == "__main__":
    main()


"""

**************parse input/output*****************
Input info:
        name:      input0
        data Type: 1
        shape:     [1, 3, 64, 64]
Output info:
        name:      input0
        data Type: 1
        shape:     [1, 3, 64, 64]

**************parse node************************
node info:
        name:      conv2d_1
        op_type:   Conv
        inputs:    ['input0', 'conv2d_1.weight', 'conv2d_1.bias']
        outputs:   ['conv2d_1.output']
node info:
        name:      batchNorm1
        op_type:   BatchNormalization
        inputs:    ['conv2d_1.output', 'batchNorm1.scale', 'batchNorm1.bias', 'batchNorm1.mean', 'batchNorm1.var']
        outputs:   ['batchNorm1.output']
node info:
        name:      relu1
        op_type:   Relu
        inputs:    ['batchNorm1.output']
        outputs:   ['relu1.output']
node info:
        name:      avg_pool1
        op_type:   GlobalAveragePool
        inputs:    ['relu1.output']
        outputs:   ['avg_pool1.output']
node info:
        name:      conv2d_2
        op_type:   Conv
        inputs:    ['avg_pool1.output', 'conv2d_2.weight', 'conv2d_2.bias']
        outputs:   ['output0']

**************parse initializer*****************
initializer info:
        name:      conv2d_1.weight
        data_type: 1
        shape:     [32, 3, 3, 3]
initializer info:
        name:      conv2d_1.bias
        data_type: 1
        shape:     [32]
initializer info:
        name:      batchNorm1.scale
        data_type: 1
        shape:     [32]
initializer info:
        name:      batchNorm1.bias
        data_type: 1
        shape:     [32]
initializer info:
        name:      batchNorm1.mean
        data_type: 1
        shape:     [32]
initializer info:
        name:      batchNorm1.var
        data_type: 1
        shape:     [32]
initializer info:
        name:      conv2d_2.weight
        data_type: 1
        shape:     [16, 32, 1, 1]
initializer info:
        name:      conv2d_2.bias
        data_type: 1
        shape:     [16]
"""

