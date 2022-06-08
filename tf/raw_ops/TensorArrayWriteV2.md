description: Deprecated. Use TensorArrayGradV3
robots: noindex

# tf.raw_ops.TensorArrayWriteV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Deprecated. Use TensorArrayGradV3

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TensorArrayWriteV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TensorArrayWriteV2(
    handle, index, value, flow_in, name=None
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
`index`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
A `Tensor`.
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
A `Tensor` of type `float32`.
</td>
</tr>

</table>

