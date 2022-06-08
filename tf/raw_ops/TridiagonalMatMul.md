description: Calculate product with tridiagonal matrix.
robots: noindex

# tf.raw_ops.TridiagonalMatMul

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Calculate product with tridiagonal matrix.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TridiagonalMatMul`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TridiagonalMatMul(
    superdiag, maindiag, subdiag, rhs, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Calculates product of two matrices, where left matrix is a tridiagonal matrix.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`superdiag`
</td>
<td>
A `Tensor`. Must be one of the following types: `float64`, `float32`, `complex64`, `complex128`.
Tensor of shape `[..., 1, M]`, representing superdiagonals of
tri-diagonal matrices to the left of multiplication. Last element is ignored.
</td>
</tr><tr>
<td>
`maindiag`
</td>
<td>
A `Tensor`. Must have the same type as `superdiag`.
Tensor of shape `[..., 1, M]`, representing main diagonals of tri-diagonal
matrices to the left of multiplication.
</td>
</tr><tr>
<td>
`subdiag`
</td>
<td>
A `Tensor`. Must have the same type as `superdiag`.
Tensor of shape `[..., 1, M]`, representing subdiagonals of tri-diagonal
matrices to the left of multiplication. First element is ignored.
</td>
</tr><tr>
<td>
`rhs`
</td>
<td>
A `Tensor`. Must have the same type as `superdiag`.
Tensor of shape `[..., M, N]`, representing MxN matrices to the right of
multiplication.
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
A `Tensor`. Has the same type as `superdiag`.
</td>
</tr>

</table>

