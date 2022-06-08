description: Return the ceiling of the input, element-wise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.ceil" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.ceil

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Return the ceiling of the input, element-wise.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.ceil`, `tf.compat.v1.math.ceil`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.ceil(
    x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### For example:



```
>>> tf.math.ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
<tf.Tensor: shape=(7,), dtype=float32,
numpy=array([-1., -1., -0.,  1.,  2.,  2.,  2.], dtype=float32)>
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
A <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a>. Must be one of the following types: `bfloat16`, `half`,
`float32`, `float64`. `int32`
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
A <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a>. Has the same type as `x`.
</td>
</tr>

</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">numpy compatibility</h2>

Equivalent to np.ceil


 </devsite-expandable></section>

