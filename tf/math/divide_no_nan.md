description: Computes a safe divide which returns 0 if y (denominator) is zero.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.divide_no_nan" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.divide_no_nan

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Computes a safe divide which returns 0 if `y` (denominator) is zero.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.div_no_nan`, `tf.compat.v1.math.divide_no_nan`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.divide_no_nan(
    x, y, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### For example:



```
>>> tf.constant(3.0) / 0.0
<tf.Tensor: shape=(), dtype=float32, numpy=inf>
>>> tf.math.divide_no_nan(3.0, 0.0)
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
```

Note that 0 is returned if `y` is 0 even if `x` is nonfinite:

```
>>> tf.math.divide_no_nan(np.nan, 0.0)
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
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
A `Tensor`. Must be one of the following types: `float32`, `float64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor` whose dtype is compatible with `x`.
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
The element-wise value of the x divided by y.
</td>
</tr>

</table>

