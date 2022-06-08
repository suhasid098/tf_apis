description: Outputs deterministic pseudorandom random integers from a uniform distribution.
robots: noindex

# tf.raw_ops.StatelessRandomUniformInt

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Outputs deterministic pseudorandom random integers from a uniform distribution.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.StatelessRandomUniformInt`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.StatelessRandomUniformInt(
    shape, seed, minval, maxval, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The generated values follow a uniform distribution in the range `[minval, maxval)`.

The outputs are a deterministic function of `shape`, `seed`, `minval`, and `maxval`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
The shape of the output tensor.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
2 seeds (shape [2]).
</td>
</tr><tr>
<td>
`minval`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
Minimum value (inclusive, scalar).
</td>
</tr><tr>
<td>
`maxval`
</td>
<td>
A `Tensor`. Must have the same type as `minval`.
Maximum value (exclusive, scalar).
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
A `Tensor`. Has the same type as `minval`.
</td>
</tr>

</table>

