description: Converts one or more images from YUV to RGB.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.yuv_to_rgb" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.yuv_to_rgb

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Converts one or more images from YUV to RGB.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.yuv_to_rgb`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.yuv_to_rgb(
    images
)
</code></pre>



<!-- Placeholder for "Used in" -->

Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels.
The output is only well defined if the Y value in images are in [0,1],
U and V value are in [-0.5,0.5].

As per the above description, you need to scale your YUV images if their
pixel values are not in the required range. Below given example illustrates
preprocessing of each channel of images before feeding them to `yuv_to_rgb`.

```python
yuv_images = tf.random.uniform(shape=[100, 64, 64, 3], maxval=255)
last_dimension_axis = len(yuv_images.shape) - 1
yuv_tensor_images = tf.truediv(
    tf.subtract(
        yuv_images,
        tf.reduce_min(yuv_images)
    ),
    tf.subtract(
        tf.reduce_max(yuv_images),
        tf.reduce_min(yuv_images)
     )
)
y, u, v = tf.split(yuv_tensor_images, 3, axis=last_dimension_axis)
target_uv_min, target_uv_max = -0.5, 0.5
u = u * (target_uv_max - target_uv_min) + target_uv_min
v = v * (target_uv_max - target_uv_min) + target_uv_min
preprocessed_yuv_images = tf.concat([y, u, v], axis=last_dimension_axis)
rgb_tensor_images = tf.image.yuv_to_rgb(preprocessed_yuv_images)
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
2-D or higher rank. Image data to convert. Last dimension must be
size 3.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`images`
</td>
<td>
tensor with the same shape as `images`.
</td>
</tr>
</table>

