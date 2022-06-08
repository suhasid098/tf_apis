description: Creates a tensor with all elements set to one (1).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.ones" />
<meta itemprop="path" content="Stable" />
</div>

# tf.ones

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Creates a tensor with all elements set to one (1).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.ones`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.ones(
    shape,
    dtype=<a href="../tf/dtypes.md#float32"><code>tf.dtypes.float32</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

See also <a href="../tf/ones_like.md"><code>tf.ones_like</code></a>, <a href="../tf/zeros.md"><code>tf.zeros</code></a>, <a href="../tf/fill.md"><code>tf.fill</code></a>, <a href="../tf/eye.md"><code>tf.eye</code></a>.

This operation returns a tensor of type `dtype` with shape `shape` and
all elements set to one.

```
>>> tf.ones([3, 4], tf.int32)
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]], dtype=int32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`
</td>
<td>
A `list` of integers, a `tuple` of integers, or
a 1-D `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Optional DType of an element in the resulting `Tensor`. Default is
<a href="../tf.md#float32"><code>tf.float32</code></a>.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional string. A name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` with all elements set to one (1).
</td>
</tr>

</table>

