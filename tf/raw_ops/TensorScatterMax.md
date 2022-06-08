description: Apply a sparse update to a tensor taking the element-wise maximum.
robots: noindex

# tf.raw_ops.TensorScatterMax

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Apply a sparse update to a tensor taking the element-wise maximum.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TensorScatterMax`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TensorScatterMax(
    tensor, indices, updates, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns a new tensor copied from `tensor` whose values are element-wise maximum between
tensor and updates according to the indices.

```
>>> tensor = [0, 0, 0, 0, 0, 0, 0, 0] 
>>> indices = [[1], [4], [5]]
>>> updates = [1, -1, 1]
>>> tf.tensor_scatter_nd_max(tensor, indices, updates).numpy()
array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int32)
```

Refer to <a href="../../tf/tensor_scatter_nd_update.md"><code>tf.tensor_scatter_nd_update</code></a> for more details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
A `Tensor`. Tensor to update.
</td>
</tr><tr>
<td>
`indices`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
Index tensor.
</td>
</tr><tr>
<td>
`updates`
</td>
<td>
A `Tensor`. Must have the same type as `tensor`.
Updates to scatter into output.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `tensor`.
</td>
</tr>

</table>

