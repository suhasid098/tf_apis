description: Creates a dataset that applies f to the outputs of input_dataset.
robots: noindex

# tf.raw_ops.FlatMapDataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a dataset that applies `f` to the outputs of `input_dataset`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.FlatMapDataset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.FlatMapDataset(
    input_dataset,
    other_arguments,
    f,
    output_types,
    output_shapes,
    metadata=&#x27;&#x27;,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
Dataset variant, and FlatMapDataset will flatten successive results
into a single Dataset.

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
`other_arguments`
</td>
<td>
A list of `Tensor` objects.
</td>
</tr><tr>
<td>
`f`
</td>
<td>
A function decorated with @Defun.
A function mapping elements of `input_dataset`, concatenated with
`other_arguments`, to a Dataset variant that contains elements matching
`output_types` and `output_shapes`.
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

