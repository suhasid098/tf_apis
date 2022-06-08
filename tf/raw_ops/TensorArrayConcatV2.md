description: Deprecated. Use TensorArrayConcatV3
robots: noindex

# tf.raw_ops.TensorArrayConcatV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Deprecated. Use TensorArrayConcatV3

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TensorArrayConcatV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TensorArrayConcatV2(
    handle, flow_in, dtype, element_shape_except0=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`handle`
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`flow_in`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>.
</td>
</tr><tr>
<td>
`element_shape_except0`
</td>
<td>
An optional <a href="../../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `ints`. Defaults to `None`.
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
A tuple of `Tensor` objects (value, lengths).
</td>
</tr>
<tr>
<td>
`value`
</td>
<td>
A `Tensor` of type `dtype`.
</td>
</tr><tr>
<td>
`lengths`
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr>
</table>

