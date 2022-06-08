description: Converts one or more images from Grayscale to RGB.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.grayscale_to_rgb" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.grayscale_to_rgb

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Converts one or more images from Grayscale to RGB.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.grayscale_to_rgb`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.grayscale_to_rgb(
    images, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Outputs a tensor of the same `DType` and rank as `images`.  The size of the
last dimension of the output is 3, containing the RGB value of the pixels.
The input images' last dimension must be size 1.

```
>>> original = tf.constant([[[1.0], [2.0], [3.0]]])
>>> converted = tf.image.grayscale_to_rgb(original)
>>> print(converted.numpy())
[[[1. 1. 1.]
  [2. 2. 2.]
  [3. 3. 3.]]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`images`
</td>
<td>
The Grayscale tensor to convert. The last dimension must be size 1.
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
The converted grayscale image(s).
</td>
</tr>

</table>

