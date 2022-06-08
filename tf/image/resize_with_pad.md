description: Resizes and pads an image to a target width and height.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.resize_with_pad" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.resize_with_pad

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Resizes and pads an image to a target width and height.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.resize_with_pad(
    image,
    target_height,
    target_width,
    method=ResizeMethod.BILINEAR,
    antialias=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

Resizes an image to a target width and height by keeping
the aspect ratio the same without distortion. If the target
dimensions don't match the image dimensions, the image
is resized and then padded with zeroes to match requested
dimensions.

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
</tr><tr>
<td>
`method`
</td>
<td>
Method to use for resizing image. See <a href="../../tf/image/resize.md"><code>image.resize()</code></a>
</td>
</tr><tr>
<td>
`antialias`
</td>
<td>
Whether to use anti-aliasing when resizing. See 'image.resize()'.
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
Resized and padded image.
If `images` was 4-D, a 4-D float Tensor of shape
`[batch, new_height, new_width, channels]`.
If `images` was 3-D, a 3-D float Tensor of shape
`[new_height, new_width, channels]`.
</td>
</tr>

</table>

