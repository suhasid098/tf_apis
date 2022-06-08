description: Scatters updates into a tensor of shape shape according to indices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.scatter_nd" />
<meta itemprop="path" content="Stable" />
</div>

# tf.scatter_nd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Scatters `updates` into a tensor of shape `shape` according to `indices`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.manip.scatter_nd`, `tf.compat.v1.scatter_nd`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.scatter_nd(
    indices, updates, shape, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Update the input tensor by scattering sparse `updates` according to individual values at the specified `indices`.
This op returns an `output` tensor with the `shape` you specify. This op is the
inverse of the <a href="../tf/gather_nd.md"><code>tf.gather_nd</code></a> operator which extracts values or slices from a
given tensor.

This operation is similar to <a href="../tf/tensor_scatter_nd_add.md"><code>tf.tensor_scatter_nd_add</code></a>, except that the tensor
is zero-initialized. Calling <a href="../tf/scatter_nd.md"><code>tf.scatter_nd(indices, values, shape)</code></a>
is identical to calling
`tf.tensor_scatter_nd_add(tf.zeros(shape, values.dtype), indices, values)`

If `indices` contains duplicates, the duplicate `values` are accumulated
(summed).

**WARNING**: The order in which updates are applied is nondeterministic, so the
output will be nondeterministic if `indices` contains duplicates;
numbers summed in different order may yield different results because of some
numerical approximation issues.

`indices` is an integer tensor of shape `shape`. The last dimension
of `indices` can be at most the rank of `shape`:

    indices.shape[-1] <= shape.rank

The last dimension of `indices` corresponds to indices of elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.

`updates` is a tensor with shape:

    indices.shape[:-1] + shape[indices.shape[-1]:]

The simplest form of the scatter op is to insert individual elements in
a tensor by index. Consider an example where you want to insert 4 scattered
elements in a rank-1 tensor with 8 elements.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
</div>

In Python, this scatter operation would look like this:

```python
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

    [0, 11, 0, 10, 9, 0, 0, 12]

You can also insert entire slices of a higher rank tensor all at once. For
example, you can insert two slices in the first dimension of a rank-3 tensor
with two matrices of new values.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
</div>

In Python, this scatter operation would look like this:

```python
    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`indices`
</td>
<td>
A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
Tensor of indices.
</td>
</tr><tr>
<td>
`updates`
</td>
<td>
A `Tensor`. Values to scatter into the output tensor.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A `Tensor`. Must have the same type as `indices`.
1-D. The shape of the output tensor.
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
A `Tensor`. Has the same type as `updates`.
</td>
</tr>

</table>

