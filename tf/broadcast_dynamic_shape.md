description: Computes the shape of a broadcast given symbolic shapes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.broadcast_dynamic_shape" />
<meta itemprop="path" content="Stable" />
</div>

# tf.broadcast_dynamic_shape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Computes the shape of a broadcast given symbolic shapes.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.broadcast_dynamic_shape`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.broadcast_dynamic_shape(
    shape_x, shape_y
)
</code></pre>



<!-- Placeholder for "Used in" -->

When `shape_x` and `shape_y` are Tensors representing shapes (i.e. the result
of calling tf.shape on another Tensor) this computes a Tensor which is the
shape of the result of a broadcasting op applied in tensors of shapes
`shape_x` and `shape_y`.

This is useful when validating the result of a broadcasting operation when the
tensors do not have statically known shapes.

#### Example:



```
>>> shape_x = (1, 2, 3)
>>> shape_y = (5, 1, 3)
>>> tf.broadcast_dynamic_shape(shape_x, shape_y)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([5, 2, 3], ...>
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
A rank 1 integer `Tensor`, representing the shape of x.
</td>
</tr><tr>
<td>
`shape_y`
</td>
<td>
A rank 1 integer `Tensor`, representing the shape of y.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A rank 1 integer `Tensor` representing the broadcasted shape.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`InvalidArgumentError`
</td>
<td>
If the two shapes are incompatible for
broadcasting.
</td>
</tr>
</table>

