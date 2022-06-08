description: Compute inverse error function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.erfinv" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.erfinv

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Compute inverse error function.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.erfinv`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.erfinv(
    x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given `x`, compute the inverse error function of `x`. This function
is the inverse of <a href="../../tf/math/erf.md"><code>tf.math.erf</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
`Tensor` with type `float` or `double`.
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
Inverse error function of `x`.
</td>
</tr>

</table>

