# Folding Constants

## Introduction


Constant folding involves pre-computing expressions that do not depend on runtime
information. Practically, this would mean that any nodes that are dependent only on
`Constant`s in an ONNX GraphSurgeon graph can be folded.

**One limitation of ONNX GraphSurgeon's built-in constant folding is that it will not**
**rotate nodes.(没有交换律) So, assuming `x` is a graph input and `c0`, `c1` and `c2` are constants in**
**the graph:**

- `x + (c0 + (c1 + c2))` **will** be folded
- `((x + c0) + c1) + c2` will **not** be folded, even though this is mathematically equivalent
  (when not considering floating point rounding error) to the previous expression.

## Prerequisites

1. ONNX GraphSurgeon uses [ONNX Runtime](https://github.com/microsoft/onnxruntime) to
   evaluate the constant expressions in the graph. This can be installed with:
   ```bash
   python3 -m pip install onnxruntime
   ```

## Running the example

1. Generate a model with several nodes and save it to `model.onnx` by running:

   ```bash
   python3 generate.py
   ```
   The generated model computes `output = input + ((a + b) + d)` where `a`,`b`, and `d` are constants
   all set to `1`:

   ![../resources/05_model.onnx.png](../resources/05_model.onnx.png)
2. Fold constants in the graph, and save it to `folded.onnx` by running:

   ```bash
   python3 fold.py
   ```
   This will replace the expression: `((a + b) + d)` with a single constant tensor (which will be all `3`s).
   The resulting graph will compute `output = input + e` where `e = ((a + b) + d)`:

   This script will also display the help output for `Graph.fold_constants()`

   ![../resources/05_folded.onnx.png](../resources/05_folded.onnx.png)

- before

![image-20240220105431408](./assets/image-20240220105431408.png)

- after

```python
# 实做的时候发现input:x2并没有删去,要加下面这段代码
fisrtMul = [node for node in graph.nodes if node.op == "Mul"][0]
graph.inputs = [inp for inp in fisrtMul.inputs if inp.name != "a"]
```

![image-20240220105458501](./assets/image-20240220105458501.png)
