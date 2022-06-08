description: Randomly flip an image horizontally (left to right).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.random_flip_left_right" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.random_flip_left_right

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Randomly flip an image horizontally (left to right).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.random_flip_left_right`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.random_flip_left_right(
    image, seed=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

With a 1 in 2 chance, outputs the contents of `image` flipped along the
second dimension, which is `width`.  Otherwise output the image as-is.
When passing a batch of images, each image will be randomly flipped
independent of other images.

#### Example usage:



```
>>> image = np.array([[[1], [2]], [[3], [4]]])
>>> tf.image.random_flip_left_right(image, 5).numpy().tolist()
[[[2], [1]], [[4], [3]]]
```

Randomly flip multiple images.

```
>>> images = np.array(
... [
...     [[[1], [2]], [[3], [4]]],
...     [[[5], [6]], [[7], [8]]]
... ])
>>> tf.image.random_flip_left_right(images, 6).numpy().tolist()
[[[[2], [1]], [[4], [3]]], [[[5], [6]], [[7], [8]]]]
```

For producing deterministic results given a `seed` value, use
<a href="../../tf/image/stateless_random_flip_left_right.md"><code>tf.image.stateless_random_flip_left_right</code></a>. Unlike using the `seed` param
with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
same results given the same seed independent of how many times the function is
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
4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
of shape `[height, width, channels]`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A Python integer. Used to create a random seed. See
<a href="../../tf/compat/v1/set_random_seed.md"><code>tf.compat.v1.set_random_seed</code></a> for behavior.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of the same type and shape as `image`.
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
if the shape of `image` not supported.
</td>
</tr>
</table>

