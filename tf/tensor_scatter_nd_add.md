description: Adds sparse updates to an existing tensor according to indices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tensor_scatter_nd_add" />
<meta itemprop="path" content="Stable" />
</div>

# tf.tensor_scatter_nd_add

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Adds sparse `updates` to an existing tensor according to `indices`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tensor_scatter_add`, `tf.compat.v1.tensor_scatter_nd_add`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tensor_scatter_nd_add(
    tensor, indices, updates, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation creates a new tensor by adding sparse `updates` to the passed
in `tensor`.
This operation is very similar to <a href="../tf/compat/v1/scatter_nd_add.md"><code>tf.compat.v1.scatter_nd_add</code></a>, except that the
updates are added onto an existing tensor (as opposed to a variable). If the
memory for the existing tensor cannot be re-used, a copy is made and updated.

`indices` is an integer tensor containing indices into a new tensor of shape
`tensor.shape`.  The last dimension of `indices` can be at most the rank of
`tensor.shape`:

```
indices.shape[-1] <= tensor.shape.rank
```

The last dimension of `indices` corresponds to indices into elements
(if `indices.shape[-1] = tensor.shape.rank`) or slices
(if `indices.shape[-1] < tensor.shape.rank`) along dimension
`indices.shape[-1]` of `tensor.shape`.  `updates` is a tensor with shape

```
indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
```

The simplest form of `tensor_scatter_nd_add` is to add individual elements to a
tensor by index. For example, say we want to add 4 elements in a rank-1
tensor with 8 elements.

In Python, this scatter add operation would look like this:

```
>>> indices = tf.constant([[4], [3], [1], [7]])
>>> updates = tf.constant([9, 10, 11, 12])
>>> tensor = tf.ones([8], dtype=tf.int32)
>>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
>>> updated
<tf.Tensor: shape=(8,), dtype=int32,
numpy=array([ 1, 12,  1, 11, 10,  1,  1, 13], dtype=int32)>
```

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

In Python, this scatter add operation would look like this:

```
>>> indices = tf.constant([[0], [2]])
>>> updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
...                         [7, 7, 7, 7], [8, 8, 8, 8]],
...                        [[5, 5, 5, 5], [6, 6, 6, 6],
...                         [7, 7, 7, 7], [8, 8, 8, 8]]])
>>> tensor = tf.ones([4, 4, 4],dtype=tf.int32)
>>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
>>> updated
<tf.Tensor: shape=(4, 4, 4), dtype=int32,
numpy=array([[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
             [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
             [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
             [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=int32)>
```

Note: on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
A `Tensor`. Tensor to copy/update.
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

