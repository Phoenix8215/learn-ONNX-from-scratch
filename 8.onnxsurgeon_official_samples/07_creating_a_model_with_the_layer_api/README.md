# Creating A Model Using The Graph Layer API

## Introduction

This example uses the `Graph.layer()` function in conjunction with `Graph.register()` to
demonstrate **how to construct complicated ONNX models more easily.**

**The `Graph.layer()` API allows you to more easily add `Node`s to a `Graph`. In addition to**
**creating nodes, this function can also create the input and output tensors, and automatically**
**insert the node in the graph. For details, see the `help` output for `Graph.layer()`.**

**Note**: You still need to set `Graph` inputs and outputs yourself!

`Graph.layer()` can be used to implement your own functions that can be registered with `Graph.register()`.
For example, to implement a `graph.add` function that inserts an "Add" operation into the graph, you could write:
```python
@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])
```

and invoke it like so:
```python
[Y] = graph.add(*graph.add(X, B), C)
```

This would add a set of nodes which compute `Y = (X + B) + C` (assuming X, B, C are some tensors in the graph)
to the graph **without requiring you to manually create the intermediate tensors involved.**

## Running the example

1. Generate a moderately complex model and save it to `model.onnx` by running:
    ```bash
    python3 generate.py
    ```

    This script will also display the help output for `Graph.layer()`

    The generated model will look like this:

    ![../resources/07_model.onnx.png](../resources/07_model.onnx.png)

---

```py
@gs.Graph.register()
def mul(self, a, b):
    return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"])


@gs.Graph.register()
def gemm(self, a, b, trans_a=False, trans_b=False):
    attrs = {"transA": int(trans_a), "transB": int(trans_b)}
    return self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs)
```

`mul`和`gemm`提供了不同层次的乘法操作，`mul`是逐元素的简单乘法，而`gemm`是更复杂的矩阵乘法操作，支持更广泛的线性代数应用，包括可选的矩阵转置功能。

---

```python
@gs.Graph.register()
def add(self, a, b):
    # The Graph.layer function creates a node, adds inputs and outputs to it, and finally adds it to the graph.
    # It returns the output tensors of the node to make it easy to chain.
    # The function will append an index to any strings provided for inputs/outputs prior
    # to using them to construct tensors. This will ensure that multiple calls to the layer() function
    # will generate distinct tensors. However, this does NOT guarantee that there will be no overlap with
    # other tensors in the graph. Hence, you should choose the prefixes to minimize the possibility of
    # collisions.
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])
```

![image-20240209140441896](./assets/image-20240209140441896.png)
