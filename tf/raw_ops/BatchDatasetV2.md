description: Creates a dataset that batches batch_size elements from input_dataset.
robots: noindex

# tf.raw_ops.BatchDatasetV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a dataset that batches `batch_size` elements from `input_dataset`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.BatchDatasetV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.BatchDatasetV2(
    input_dataset,
    batch_size,
    drop_remainder,
    output_types,
    output_shapes,
    parallel_copy=False,
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
`input_dataset`
</td>
<td>
A `Tensor` of type `variant`.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
A `Tensor` of type `int64`.
A scalar representing the number of elements to accumulate in a batch.
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
A `Tensor` of type `bool`.
A scalar representing whether the last batch should be dropped in case its size
is smaller than desired.
</td>
</tr><tr>
<td>
`output_types`
</td>
<td>
A list of `tf.DTypes` that has length `>= 1`.
</td>
</tr><tr>
<td>
`output_shapes`
</td>
<td>
A list of shapes (each a <a href="../../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `ints`) that has length `>= 1`.
</td>
</tr><tr>
<td>
`parallel_copy`
</td>
<td>
An optional `bool`. Defaults to `False`.
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

