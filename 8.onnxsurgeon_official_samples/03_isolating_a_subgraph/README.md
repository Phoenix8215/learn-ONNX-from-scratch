# Isolating A Subgraph

## Introduction

This example first generates a basic model,then extracts a subgraph from this model.

**Isolating a subgraph from a model is as simple as modifying the inputs and**
**outputs of the graph, running `graph.cleanup()`, and then re-exporting the graph.**

We do **not** need to know which nodes, initializers, or intermediate tensors we
want - marking the inputs and outputs is sufficient for ONNX GraphSurgeon to be able
to determine the other information automatically.

## Running the example

1. Generate a model with several nodes and save it to `model.onnx` by running:

   ```bash
   python3 generate.py
   ```

   The generated model computes `Y = x0 + (a * x1 + b)`:

   ![../resources/03_model.onnx.png](../resources/03_model.onnx.png)
2. Isolate the subgraph that computes `(a * x1 + b)` and save it to `subgraph.onnx` by running:

   ```bash
   python3 isolate.py
   ```

   The resulting model computes `add_out = (a * x1 + b)`:

   ![../resources/03_subgraph.onnx.png](../resources/03_subgraph.onnx.png)

---

```shell
OrderedDict([('x0', Variable (x0): (shape=[1, 3, 224, 224], dtype=float32)), ('x1', Variable (x1): (shape=[1, 3, 224, 224], dtype=float32)), ('a', Constant (a): (shape=[1, 3, 224, 224], dtype=float32)
LazyValues (shape=[1, 3, 224, 224], dtype=float32)), ('mul_out', Variable (mul_out): (shape=None, dtype=None)), ('b', Constant (b): (shape=[1, 3, 224, 224], dtype=float32)
LazyValues (shape=[1, 3, 224, 224], dtype=float32)), ('add_out', Variable (add_out): (shape=None, dtype=None)), ('Y', Variable (Y): (shape=[1, 3, 224, 224], dtype=float32))])
```

`LazyValues` 是与ONNX GraphSurgeon库相关的一个术语，它指的是在图的构造阶段，某些张量（tensor）的值是延迟计算的，也就是说，这些值并不是立即计算出来的，而是等到需要的时候才计算。这种机制在处理复杂的图转换和优化时非常有用，因为它允许图的构建者在不立即计算或确定所有张量值的情况下，构造和修改图结构。
