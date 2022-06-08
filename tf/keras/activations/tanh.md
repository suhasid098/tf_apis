description: Hyperbolic tangent activation function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.activations.tanh" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.activations.tanh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/activations.py#L354-L373">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Hyperbolic tangent activation function.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.activations.tanh`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.activations.tanh(
    x
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### For example:



```
>>> a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32)
>>> b = tf.keras.activations.tanh(a)
>>> b.numpy()
array([-0.9950547, -0.7615942,  0.,  0.7615942,  0.9950547], dtype=float32)
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
Input tensor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Tensor of same shape and dtype of input `x`, with tanh activation:
`tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))`.
</td>
</tr>

</table>

