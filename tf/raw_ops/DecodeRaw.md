description: Reinterpret the bytes of a string as a vector of numbers.
robots: noindex

# tf.raw_ops.DecodeRaw

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Reinterpret the bytes of a string as a vector of numbers.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.DecodeRaw`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.DecodeRaw(
    bytes, out_type, little_endian=True, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`bytes`
</td>
<td>
A `Tensor` of type `string`.
All the elements must have the same length.
</td>
</tr><tr>
<td>
`out_type`
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.half, tf.float32, tf.float64, tf.int32, tf.uint16, tf.uint8, tf.int16, tf.int8, tf.int64, tf.complex64, tf.complex128, tf.bool, tf.bfloat16`.
</td>
</tr><tr>
<td>
`little_endian`
</td>
<td>
An optional `bool`. Defaults to `True`.
Whether the input `bytes` are in little-endian order.
Ignored for `out_type` values that are stored in a single byte like
`uint8`.
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
A `Tensor` of type `out_type`.
</td>
</tr>

</table>

