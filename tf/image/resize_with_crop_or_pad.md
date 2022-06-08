description: Crops and/or pads an image to a target width and height.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.resize_with_crop_or_pad" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.resize_with_crop_or_pad

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Crops and/or pads an image to a target width and height.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.resize_image_with_crop_or_pad`, `tf.compat.v1.image.resize_with_crop_or_pad`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.resize_with_crop_or_pad(
    image, target_height, target_width
)
</code></pre>



<!-- Placeholder for "Used in" -->

Resizes an image to a target width and height by either centrally
cropping the image or padding it evenly with zeros.

If `width` or `height` is greater than the specified `target_width` or
`target_height` respectively, this op centrally crops along that dimension.

#### For example:



```
>>> image = np.arange(75).reshape(5, 5, 3)  # create 3-D image input
>>> image[:,:,0]  # print first channel just for demo purposes
array([[ 0,  3,  6,  9, 12],
       [15, 18, 21, 24, 27],
       [30, 33, 36, 39, 42],
       [45, 48, 51, 54, 57],
       [60, 63, 66, 69, 72]])
>>> image = tf.image.resize_with_crop_or_pad(image, 3, 3)  # crop
>>> # print first channel for demo purposes; centrally cropped output
>>> image[:,:,0]
<tf.Tensor: shape=(3, 3), dtype=int64, numpy=
array([[18, 21, 24],
       [33, 36, 39],
       [48, 51, 54]])>
```

If `width` or `height` is smaller than the specified `target_width` or
`target_height` respectively, this op centrally pads with 0 along that
dimension.

#### For example:



```
>>> image = np.arange(1, 28).reshape(3, 3, 3)  # create 3-D image input
>>> image[:,:,0]  # print first channel just for demo purposes
array([[ 1,  4,  7],
       [10, 13, 16],
       [19, 22, 25]])
>>> image = tf.image.resize_with_crop_or_pad(image, 5, 5)  # pad
>>> # print first channel for demo purposes; we should see 0 paddings
>>> image[:,:,0]
<tf.Tensor: shape=(5, 5), dtype=int64, numpy=
array([[ 0,  0,  0,  0,  0],
       [ 0,  1,  4,  7,  0],
       [ 0, 10, 13, 16,  0],
       [ 0, 19, 22, 25,  0],
       [ 0,  0,  0,  0,  0]])>
```

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
`target_height`
</td>
<td>
Target height.
</td>
</tr><tr>
<td>
`target_width`
</td>
<td>
Target width.
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
if `target_height` or `target_width` are zero or negative.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Cropped and/or padded image.
If `images` was 4-D, a 4-D float Tensor of shape
`[batch, new_height, new_width, channels]`.
If `images` was 3-D, a 3-D float Tensor of shape
`[new_height, new_width, channels]`.
</td>
</tr>

</table>

