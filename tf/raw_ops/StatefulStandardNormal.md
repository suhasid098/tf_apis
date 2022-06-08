description: Outputs random values from a normal distribution. This op is deprecated in favor of op 'StatefulStandardNormalV2'
robots: noindex

# tf.raw_ops.StatefulStandardNormal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Outputs random values from a normal distribution. This op is deprecated in favor of op 'StatefulStandardNormalV2'

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.StatefulStandardNormal`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.StatefulStandardNormal(
    resource,
    shape,
    dtype=<a href="../../tf/dtypes.md#float32"><code>tf.dtypes.float32</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The generated values will have mean 0 and standard deviation 1.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`resource`
</td>
<td>
A `Tensor` of type `resource`.
The handle of the resource variable that stores the state of the RNG.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A `Tensor`. The shape of the output tensor.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
An optional <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>. Defaults to <a href="../../tf.md#float32"><code>tf.float32</code></a>.
The type of the output.
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
A `Tensor` of type `dtype`.
</td>
</tr>

</table>

