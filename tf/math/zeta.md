description: Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.zeta" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.zeta

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.zeta`, `tf.compat.v1.zeta`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.zeta(
    x, q, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The Hurwitz zeta function is defined as:


\\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`.
</td>
</tr><tr>
<td>
`q`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
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
A `Tensor`. Has the same type as `x`.
</td>
</tr>

</table>

