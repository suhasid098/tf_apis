description: Computes the Exponential integral of x element-wise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.special.expint" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.special.expint

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/special_math_ops.py">View source</a>



Computes the Exponential integral of `x` element-wise.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.special.expint`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.special.expint(
    x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The Exponential integral is defined as the integral of `exp(t) / t` from
`-inf` to `x`, with the domain of definition all positive real numbers.

```
>>> tf.math.special.expint([1., 1.1, 2.1, 4.1]).numpy()
array([ 1.8951179,  2.1673784,  5.3332353, 21.048464], dtype=float32)
```

This implementation is based off of the Cephes math library.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor` or `SparseTensor`. Must be one of the following types:
`float32`, `float64`.
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
A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
</td>
</tr>

</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">scipy compatibility</h2>

Equivalent to scipy.special.expi


 </devsite-expandable></section>

