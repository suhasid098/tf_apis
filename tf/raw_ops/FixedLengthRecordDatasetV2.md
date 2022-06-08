robots: noindex

# tf.raw_ops.FixedLengthRecordDatasetV2

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
<p>`tf.compat.v1.raw_ops.FixedLengthRecordDatasetV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.FixedLengthRecordDatasetV2(
    filenames,
    header_bytes,
    record_bytes,
    footer_bytes,
    buffer_size,
    compression_type,
    metadata=&#x27;&#x27;,
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
`filenames`
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`header_bytes`
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`record_bytes`
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`footer_bytes`
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`buffer_size`
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`compression_type`
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`metadata`
</td>
<td>
An optional `string`. Defaults to `""`.
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
A `Tensor` of type `variant`.
</td>
</tr>

</table>

