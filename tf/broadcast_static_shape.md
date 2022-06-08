description: Computes the shape of a broadcast given known shapes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.broadcast_static_shape" />
<meta itemprop="path" content="Stable" />
</div>

# tf.broadcast_static_shape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Computes the shape of a broadcast given known shapes.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.broadcast_static_shape`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.broadcast_static_shape(
    shape_x, shape_y
)
</code></pre>



<!-- Placeholder for "Used in" -->

When `shape_x` and `shape_y` are fully known `TensorShape`s this computes a
`TensorShape` which is the shape of the result of a broadcasting op applied in
tensors of shapes `shape_x` and `shape_y`.

For example, if shape_x is `TensorShape([1, 2, 3])` and shape_y is
`TensorShape([5, 1, 3])`, the result is a TensorShape whose value is
`TensorShape([5, 2, 3])`.

This is useful when validating the result of a broadcasting operation when the
tensors have statically known shapes.

#### Example:



```
>>> shape_x = tf.TensorShape([1, 2, 3])
>>> shape_y = tf.TensorShape([5, 1 ,3])
>>> tf.broadcast_static_shape(shape_x, shape_y)
TensorShape([5, 2, 3])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape_x`
</td>
<td>
A `TensorShape`
</td>
</tr><tr>
<td>
`shape_y`
</td>
<td>
A `TensorShape`
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `TensorShape` representing the broadcasted shape.
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
If the two shapes can not be broadcasted.
</td>
</tr>
</table>

