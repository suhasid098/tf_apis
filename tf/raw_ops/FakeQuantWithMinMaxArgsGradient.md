description: Compute gradients for a FakeQuantWithMinMaxArgs operation.
robots: noindex

# tf.raw_ops.FakeQuantWithMinMaxArgsGradient

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Compute gradients for a FakeQuantWithMinMaxArgs operation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.FakeQuantWithMinMaxArgsGradient`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.FakeQuantWithMinMaxArgsGradient(
    gradients, inputs, min=-6, max=6, num_bits=8, narrow_range=False, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`gradients`
</td>
<td>
A `Tensor` of type `float32`.
Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
</td>
</tr><tr>
<td>
`inputs`
</td>
<td>
A `Tensor` of type `float32`.
Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
</td>
</tr><tr>
<td>
`min`
</td>
<td>
An optional `float`. Defaults to `-6`.
</td>
</tr><tr>
<td>
`max`
</td>
<td>
An optional `float`. Defaults to `6`.
</td>
</tr><tr>
<td>
`num_bits`
</td>
<td>
An optional `int`. Defaults to `8`.
</td>
</tr><tr>
<td>
`narrow_range`
</td>
<td>
An optional `bool`. Defaults to `False`.
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

