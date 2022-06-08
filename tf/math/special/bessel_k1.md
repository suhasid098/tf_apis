description: Computes the Bessel k1 function of x element-wise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.special.bessel_k1" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.special.bessel_k1

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/special_math_ops.py">View source</a>



Computes the Bessel k1 function of `x` element-wise.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.special.bessel_k1`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.special.bessel_k1(
    x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Modified Bessel function of order 1.

It is preferable to use the numerically stabler function `k1e(x)` instead.

```
>>> tf.math.special.bessel_k1([0.5, 1., 2., 4.]).numpy()
array([1.65644112, 0.60190723, 0.13986588, 0.0124835 ], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
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

Equivalent to scipy.special.k1


 </devsite-expandable></section>

