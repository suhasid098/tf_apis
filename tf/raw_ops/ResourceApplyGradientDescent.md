description: Update '*var' by subtracting 'alpha' * 'delta' from it.
robots: noindex

# tf.raw_ops.ResourceApplyGradientDescent

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Update '*var' by subtracting 'alpha' * 'delta' from it.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.ResourceApplyGradientDescent`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.ResourceApplyGradientDescent(
    var, alpha, delta, use_locking=False, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`var`
</td>
<td>
A `Tensor` of type `resource`. Should be from a Variable().
</td>
</tr><tr>
<td>
`alpha`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
Scaling factor. Must be a scalar.
</td>
</tr><tr>
<td>
`delta`
</td>
<td>
A `Tensor`. Must have the same type as `alpha`. The change.
</td>
</tr><tr>
<td>
`use_locking`
</td>
<td>
An optional `bool`. Defaults to `False`.
If `True`, the subtraction will be protected by a lock;
otherwise the behavior is undefined, but may exhibit less contention.
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

