robots: noindex

# tf.raw_ops.QuantizedConv2DAndReluAndRequantize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>





<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizedConv2DAndReluAndRequantize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizedConv2DAndReluAndRequantize(
    input,
    filter,
    min_input,
    max_input,
    min_filter,
    max_filter,
    min_freezed_output,
    max_freezed_output,
    strides,
    padding,
    out_type=<a href="../../tf/dtypes.md#quint8"><code>tf.dtypes.quint8</code></a>,
    dilations=[1, 1, 1, 1],
    padding_list=[],
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
</td>
</tr><tr>
<td>
`filter`
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
</td>
</tr><tr>
<td>
`min_input`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_input`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`min_filter`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_filter`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`min_freezed_output`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_freezed_output`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`strides`
</td>
<td>
A list of `ints`.
</td>
</tr><tr>
<td>
`padding`
</td>
<td>
A `string` from: `"SAME", "VALID"`.
</td>
</tr><tr>
<td>
`out_type`
</td>
<td>
An optional <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to <a href="../../tf.md#quint8"><code>tf.quint8</code></a>.
</td>
</tr><tr>
<td>
`dilations`
</td>
<td>
An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
</td>
</tr><tr>
<td>
`padding_list`
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
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
A tuple of `Tensor` objects (output, min_output, max_output).
</td>
</tr>
<tr>
<td>
`output`
</td>
<td>
A `Tensor` of type `out_type`.
</td>
</tr><tr>
<td>
`min_output`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_output`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

