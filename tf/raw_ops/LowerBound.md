description: Applies lower_bound(sorted_search_values, values) along each row.
robots: noindex

# tf.raw_ops.LowerBound

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Applies lower_bound(sorted_search_values, values) along each row.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.LowerBound`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.LowerBound(
    sorted_inputs,
    values,
    out_type=<a href="../../tf/dtypes.md#int32"><code>tf.dtypes.int32</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Each set of rows with the same index in (sorted_inputs, values) is treated
independently.  The resulting row is the equivalent of calling
`np.searchsorted(sorted_inputs, values, side='left')`.

The result is not a global index to the entire
`Tensor`, but rather just the index in the last dimension.

A 2-D example:
  sorted_sequence = [[0, 3, 9, 9, 10],
                     [1, 2, 3, 4, 5]]
  values = [[2, 4, 9],
            [0, 2, 6]]

  result = LowerBound(sorted_sequence, values)

  result == [[1, 2, 2],
             [0, 1, 5]]

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sorted_inputs`
</td>
<td>
A `Tensor`. 2-D Tensor where each row is ordered.
</td>
</tr><tr>
<td>
`values`
</td>
<td>
A `Tensor`. Must have the same type as `sorted_inputs`.
2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
the values that will be searched for in `sorted_search_values`.
</td>
</tr><tr>
<td>
`out_type`
</td>
<td>
An optional <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.int32, tf.int64`. Defaults to <a href="../../tf.md#int32"><code>tf.int32</code></a>.
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

