description: Softmax converts a vector of values to a probability distribution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.activations.softmax" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.activations.softmax

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/activations.py#L38-L92">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Softmax converts a vector of values to a probability distribution.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.activations.softmax`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.activations.softmax(
    x, axis=-1
)
</code></pre>



<!-- Placeholder for "Used in" -->

The elements of the output vector are in range (0, 1) and sum to 1.

Each vector is handled independently. The `axis` argument sets which axis
of the input the function is applied along.

Softmax is often used as the activation for the last
layer of a classification network because the result could be interpreted as
a probability distribution.

The softmax of each vector x is computed as
`exp(x) / tf.reduce_sum(exp(x))`.

The input values in are the log-odds of the resulting probability.

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
</tr><tr>
<td>
`axis`
</td>
<td>
Integer, axis along which the softmax normalization is applied.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Tensor, output of softmax transformation (all values are non-negative
and sum to 1).
</td>
</tr>

</table>



#### Examples:



**Example 1: standalone usage**

```
>>> inputs = tf.random.normal(shape=(32, 10))
>>> outputs = tf.keras.activations.softmax(inputs)
>>> tf.reduce_sum(outputs[0, :])  # Each sample in the batch now sums to 1
<tf.Tensor: shape=(), dtype=float32, numpy=1.0000001>
```

**Example 2: usage in a `Dense` layer**

```
>>> layer = tf.keras.layers.Dense(32, activation=tf.keras.activations.softmax)
```