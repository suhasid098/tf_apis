description: Returns the complex conjugate of a complex number.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.conj" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.conj

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Returns the complex conjugate of a complex number.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.conj`, `tf.compat.v1.math.conj`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.conj(
    x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given a tensor `x` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `x`. The
complex numbers in `x` must be of the form \\(a + bj\\), where `a` is the
real part and `b` is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

#### For example:



```
>>> x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
>>> tf.math.conj(x)
<tf.Tensor: shape=(2,), dtype=complex128,
numpy=array([-2.25-4.75j,  3.25-5.75j])>
```

If `x` is real, it is returned unchanged.

#### For example:



```
>>> x = tf.constant([-2.25, 3.25])
>>> tf.math.conj(x)
<tf.Tensor: shape=(2,), dtype=float32,
numpy=array([-2.25,  3.25], dtype=float32)>
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
`Tensor` to conjugate.  Must have numeric or variant type.
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
A `Tensor` that is the conjugate of `x` (with the same type).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If `x` is not a numeric tensor.
</td>
</tr>
</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">numpy compatibility</h2>

Equivalent to numpy.conj.


 </devsite-expandable></section>

