description: Adjust the hue of RGB images by a random factor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.random_hue" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.random_hue

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Adjust the hue of RGB images by a random factor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.random_hue`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.random_hue(
    image, max_delta, seed=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Equivalent to `adjust_hue()` but uses a `delta` randomly
picked in the interval `[-max_delta, max_delta)`.

`max_delta` must be in the interval `[0, 0.5]`.

#### Usage Example:



```
>>> x = [[[1.0, 2.0, 3.0],
...       [4.0, 5.0, 6.0]],
...     [[7.0, 8.0, 9.0],
...       [10.0, 11.0, 12.0]]]
>>> tf.image.random_hue(x, 0.2)
<tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>
```

For producing deterministic results given a `seed` value, use
<a href="../../tf/image/stateless_random_hue.md"><code>tf.image.stateless_random_hue</code></a>. Unlike using the `seed` param with
`tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the same
results given the same seed independent of how many times the function is
called, and independent of global seed settings (e.g. tf.random.set_seed).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`image`
</td>
<td>
RGB image or images. The size of the last dimension must be 3.
</td>
</tr><tr>
<td>
`max_delta`
</td>
<td>
float. The maximum value for the random delta.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
An operation-specific seed. It will be used in conjunction with the
graph-level seed to determine the real seeds that will be used in this
operation. Please see the documentation of set_random_seed for its
interaction with the graph-level random seed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Adjusted image(s), same shape and DType as `image`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if `max_delta` is invalid.
</td>
</tr>
</table>

