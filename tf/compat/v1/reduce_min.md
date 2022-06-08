description: Computes the <a href="../../../tf/math/minimum.md"><code>tf.math.minimum</code></a> of elements across dimensions of a tensor. (deprecated arguments)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.reduce_min" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.reduce_min

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Computes the <a href="../../../tf/math/minimum.md"><code>tf.math.minimum</code></a> of elements across dimensions of a tensor. (deprecated arguments)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.reduce_min`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.reduce_min(
    input_tensor,
    axis=None,
    keepdims=None,
    name=None,
    reduction_indices=None,
    keep_dims=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: SOME ARGUMENTS ARE DEPRECATED: `(keep_dims)`. They will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead

This is the reduction operation for the elementwise <a href="../../../tf/math/minimum.md"><code>tf.math.minimum</code></a> op.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
of the entries in `axis`, which must be unique. If `keepdims` is true, the
reduced dimensions are retained with length 1.

If `axis` is None, all dimensions are reduced, and a
tensor with a single element is returned.

#### Usage example:


```
>>> x = tf.constant([5, 1, 2, 4])
>>> tf.reduce_min(x)
<tf.Tensor: shape=(), dtype=int32, numpy=1>
>>> x = tf.constant([-5, -1, -2, -4])
>>> tf.reduce_min(x)
<tf.Tensor: shape=(), dtype=int32, numpy=-5>
>>> x = tf.constant([4, float('nan')])
>>> tf.reduce_min(x)
<tf.Tensor: shape=(), dtype=float32, numpy=nan>
>>> x = tf.constant([float('nan'), float('nan')])
>>> tf.reduce_min(x)
<tf.Tensor: shape=(), dtype=float32, numpy=nan>
>>> x = tf.constant([float('-inf'), float('inf')])
>>> tf.reduce_min(x)
<tf.Tensor: shape=(), dtype=float32, numpy=-inf>
```


See the numpy docs for `np.amin` and `np.nanmin` behavior.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_tensor`
</td>
<td>
The tensor to reduce. Should have real numeric type.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
The dimensions to reduce. If `None` (the default), reduces all
dimensions. Must be in the range `[-rank(input_tensor),
rank(input_tensor))`.
</td>
</tr><tr>
<td>
`keepdims`
</td>
<td>
If true, retains reduced dimensions with length 1.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr><tr>
<td>
`reduction_indices`
</td>
<td>
The old (deprecated) name for axis.
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
The reduced tensor.
</td>
</tr>

</table>

