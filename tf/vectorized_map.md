description: Parallel map on the list of tensors unpacked from elems on dimension 0.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.vectorized_map" />
<meta itemprop="path" content="Stable" />
</div>

# tf.vectorized_map

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/parallel_for/control_flow_ops.py">View source</a>



Parallel map on the list of tensors unpacked from `elems` on dimension 0.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.vectorized_map`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.vectorized_map(
    fn, elems, fallback_to_while_loop=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

This method works similar to <a href="../tf/map_fn.md"><code>tf.map_fn</code></a> but is optimized to run much faster,
possibly with a much larger memory footprint. The speedups are obtained by
vectorization (see [Auto-Vectorizing TensorFlow Graphs: Jacobians,
Auto-Batching and Beyond](https://arxiv.org/pdf/1903.04243.pdf)). The idea
behind vectorization is to semantically launch all the invocations of `fn` in
parallel and fuse corresponding operations across all these invocations. This
fusion is done statically at graph generation time and the generated code is
often similar in performance to a manually fused version.

Because <a href="../tf/vectorized_map.md"><code>tf.vectorized_map</code></a> fully parallelizes the batch, this method will
generally be significantly faster than using <a href="../tf/map_fn.md"><code>tf.map_fn</code></a>, especially in eager
mode. However this is an experimental feature and currently has a lot of
limitations:
  - There should be no data dependency between the different semantic
    invocations of `fn`, i.e. it should be safe to map the elements of the
    inputs in any order.
  - Stateful kernels may mostly not be supported since these often imply a
    data dependency. We do support a limited set of such stateful kernels
    though (like RandomFoo, Variable operations like reads, etc).
  - `fn` has limited support for control flow operations.
  - `fn` should return nested structure of Tensors or Operations. However
    if an Operation is returned, it should have zero outputs.
  - The shape and dtype of any intermediate or output tensors in the
    computation of `fn` should not depend on the input to `fn`.

#### Examples:


```python
def outer_product(a):
  return tf.tensordot(a, a, 0)

batch_size = 100
a = tf.ones((batch_size, 32, 32))
c = tf.vectorized_map(outer_product, a)
assert c.shape == (batch_size, 32, 32, 32, 32)
```

```python
# Computing per-example gradients

batch_size = 10
num_features = 32
layer = tf.keras.layers.Dense(1)

def model_fn(arg):
  with tf.GradientTape() as g:
    inp, label = arg
    inp = tf.expand_dims(inp, 0)
    label = tf.expand_dims(label, 0)
    prediction = layer(inp)
    loss = tf.nn.l2_loss(label - prediction)
  return g.gradient(loss, (layer.kernel, layer.bias))

inputs = tf.random.uniform([batch_size, num_features])
labels = tf.random.uniform([batch_size, 1])
per_example_gradients = tf.vectorized_map(model_fn, (inputs, labels))
assert per_example_gradients[0].shape == (batch_size, num_features, 1)
assert per_example_gradients[1].shape == (batch_size, 1)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fn`
</td>
<td>
The callable to be performed. It accepts one argument, which will have
the same (possibly nested) structure as `elems`, and returns a possibly
nested structure of Tensors and Operations, which may be different than
the structure of `elems`.
</td>
</tr><tr>
<td>
`elems`
</td>
<td>
A tensor or (possibly nested) sequence of tensors, each of which will
be unpacked along their first dimension. The nested sequence of the
resulting slices will be mapped over by `fn`. The first dimensions of all
elements must broadcast to a consistent value; equivalently, each
element tensor must have first dimension of either `B` or `1`, for some
common batch size `B >= 1`.
</td>
</tr><tr>
<td>
`fallback_to_while_loop`
</td>
<td>
If true, on failing to vectorize an operation,
the unsupported op is wrapped in a tf.while_loop to execute the map
iterations. Note that this fallback only happens for unsupported ops and
other parts of `fn` are still vectorized. If false, on encountering an
unsupported op, a ValueError is thrown. Note that the fallbacks can result
in slowdowns since vectorization often yields speedup of one to two orders
of magnitude.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor or (possibly nested) sequence of tensors. Each tensor packs the
results of applying fn to tensors unpacked from elems along the first
dimension, from first to last.

Although they are less common as user-visible inputs and outputs, note that
tensors of type <a href="../tf.md#variant"><code>tf.variant</code></a> which represent tensor lists (for example from
<a href="../tf/raw_ops/TensorListFromTensor.md"><code>tf.raw_ops.TensorListFromTensor</code></a>) are vectorized by stacking the list
contents rather than the variant itself, and so the container tensor will
have a scalar shape when returned rather than the usual stacked shape. This
improves the performance of control flow gradient vectorization.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If vectorization fails and fallback_to_while_loop is False.
</td>
</tr>
</table>

