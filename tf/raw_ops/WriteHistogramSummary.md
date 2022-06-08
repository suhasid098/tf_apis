description: Writes a histogram summary.
robots: noindex

# tf.raw_ops.WriteHistogramSummary

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Writes a histogram summary.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.WriteHistogramSummary`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.WriteHistogramSummary(
    writer, step, tag, values, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Writes histogram `values` at `step` with `tag` using summary `writer`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`writer`
</td>
<td>
A `Tensor` of type `resource`.
</td>
</tr><tr>
<td>
`step`
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`tag`
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`values`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `bool`.
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
The created Operation.
</td>
</tr>

</table>

