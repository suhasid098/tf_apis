description: Computes <a href="../../../tf/sparse/maximum.md"><code>tf.sparse.maximum</code></a> of elements across dimensions of a SparseTensor. (deprecated arguments) (deprecated arguments)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.sparse_reduce_max" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.sparse_reduce_max

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/sparse_ops.py">View source</a>



Computes <a href="../../../tf/sparse/maximum.md"><code>tf.sparse.maximum</code></a> of elements across dimensions of a SparseTensor. (deprecated arguments) (deprecated arguments)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.sparse.reduce_max`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.sparse_reduce_max(
    sp_input, axis=None, keepdims=None, reduction_axes=None, keep_dims=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: SOME ARGUMENTS ARE DEPRECATED: `(keep_dims)`. They will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead

Deprecated: SOME ARGUMENTS ARE DEPRECATED: `(reduction_axes)`. They will be removed in a future version.
Instructions for updating:
reduction_axes is deprecated, use axis instead

This is the reduction operation for the elementwise <a href="../../../tf/sparse/maximum.md"><code>tf.sparse.maximum</code></a> op.

This Op takes a SparseTensor and is the sparse counterpart to
<a href="../../../tf/math/reduce_max.md"><code>tf.reduce_max()</code></a>.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.

Note: A gradient is not defined for this function, so it can't be used
in training models that need gradient descent.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keepdims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
similar to the indexing rules in Python.

The values not defined in `sp_input` don't participate in the reduce max,
as opposed to be implicitly assumed 0 -- hence it can return negative values
for sparse `reduction_axes`. But, in case there are no values in
`reduction_axes`, it will reduce to 0. See second example below.

#### For example:


# 'x' represents [[1, ?, 2]
#                 [?, 3, ?]]
# where ? is implicitly-zero.

```
>>> x = tf.sparse.SparseTensor([[0, 0], [0, 2], [1, 1]], [1, 2, 3], [2, 3])
>>> tf.sparse.reduce_max(x)
<tf.Tensor: shape=(), dtype=int32, numpy=3>
>>> tf.sparse.reduce_max(x, 0)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 3, 2], dtype=int32)>
>>> tf.sparse.reduce_max(x, 1)
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 3], dtype=int32)>
>>> tf.sparse.reduce_max(x, 1, keepdims=True)
<tf.Tensor: shape=(2, 1), dtype=int32, numpy=
array([[2],
       [3]], dtype=int32)>
>>> tf.sparse.reduce_max(x, [0, 1])
<tf.Tensor: shape=(), dtype=int32, numpy=3>
```

# 'y' represents [[-7, ?]
#                 [ 4, 3]
#                 [ ?, ?]

```
>>> y = tf.sparse.SparseTensor([[0, 0,], [1, 0], [1, 1]], [-7, 4, 3],
... [3, 2])
>>> tf.sparse.reduce_max(y, 1)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([-7,  4,  0], dtype=int32)>
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sp_input`
</td>
<td>
The SparseTensor to reduce. Should have numeric type.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
The dimensions to reduce; list or scalar. If `None` (the
default), reduces all dimensions.
</td>
</tr><tr>
<td>
`keepdims`
</td>
<td>
If true, retain reduced dimensions with length 1.
</td>
</tr><tr>
<td>
`reduction_axes`
</td>
<td>
Deprecated name of `axis`.
</td>
</tr><tr>
<td>
`keep_dims`
</td>
<td>
 Deprecated alias for `keepdims`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The reduced Tensor.
</td>
</tr>

</table>

