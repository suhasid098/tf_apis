description: Draws samples from a multinomial distribution. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.multinomial" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.multinomial

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/random_ops.py">View source</a>



Draws samples from a multinomial distribution. (deprecated)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.random.multinomial`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.multinomial(
    logits, num_samples, seed=None, name=None, output_dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/random/categorical.md"><code>tf.random.categorical</code></a> instead.

#### Example:



```python
# samples has shape [1, 5], where each value is either 0 or 1 with equal
# probability.
samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`logits`
</td>
<td>
2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
`[i, :]` represents the unnormalized log-probabilities for all classes.
</td>
</tr><tr>
<td>
`num_samples`
</td>
<td>
0-D.  Number of independent samples to draw for each row slice.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A Python integer. Used to create a random seed for the distribution.
See <a href="../../../tf/random/set_seed.md"><code>tf.random.set_seed</code></a> for behavior.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the operation.
</td>
</tr><tr>
<td>
`output_dtype`
</td>
<td>
The integer type of the output: `int32` or `int64`. Defaults
to `int64`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The drawn samples of shape `[batch_size, num_samples]`.
</td>
</tr>

</table>

