description: Checks a tensor for NaN, -Inf and +Inf values.
robots: noindex

# tf.raw_ops.CheckNumericsV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Checks a tensor for NaN, -Inf and +Inf values.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.CheckNumericsV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.CheckNumericsV2(
    tensor, message, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
tensor. Unlike CheckNumerics (V1), CheckNumericsV2 distinguishes -Inf and +Inf
in the errors it throws.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
</td>
</tr><tr>
<td>
`message`
</td>
<td>
A `string`. Prefix of the error message.
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
A `Tensor`. Has the same type as `tensor`.
</td>
</tr>

</table>

