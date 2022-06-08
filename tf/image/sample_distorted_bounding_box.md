description: Generate a single randomly distorted bounding box for an image.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.sample_distorted_bounding_box" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.sample_distorted_bounding_box

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Generate a single randomly distorted bounding box for an image.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.sample_distorted_bounding_box(
    image_size,
    bounding_boxes,
    seed=0,
    min_object_covered=0.1,
    aspect_ratio_range=None,
    area_range=None,
    max_attempts=None,
    use_image_if_no_bounding_boxes=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Bounding box annotations are often supplied in addition to ground-truth labels
in image recognition or object localization tasks. A common technique for
training such a system is to randomly distort an image while preserving
its content, i.e. *data augmentation*. This Op outputs a randomly distorted
localization of an object, i.e. bounding box, given an `image_size`,
`bounding_boxes` and a series of constraints.

The output of this Op is a single bounding box that may be used to crop the
original image. The output is returned as 3 tensors: `begin`, `size` and
`bboxes`. The first 2 tensors can be fed directly into <a href="../../tf/slice.md"><code>tf.slice</code></a> to crop the
image. The latter may be supplied to <a href="../../tf/image/draw_bounding_boxes.md"><code>tf.image.draw_bounding_boxes</code></a> to
visualize what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.
The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
and the height of the underlying image.

For example,

```python
    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bounding_boxes,
        min_object_covered=0.1)

    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox_for_draw)
    tf.compat.v1.summary.image('images_with_box', image_with_box)

    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(image, begin, size)
```

Note that if no bounding box information is available, setting
`use_image_if_no_bounding_boxes = true` will assume there is a single implicit
bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
false and no bounding boxes are supplied, an error is raised.

For producing deterministic results given a `seed` value, use
<a href="../../tf/image/stateless_sample_distorted_bounding_box.md"><code>tf.image.stateless_sample_distorted_bounding_box</code></a>. Unlike using the `seed`
param with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops
guarantee the same results given the same seed independent of how many times
the function is called, and independent of global seed settings
(e.g. tf.random.set_seed).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`image_size`
</td>
<td>
A `Tensor`. Must be one of the following types: `uint8`, `int8`,
`int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.
</td>
</tr><tr>
<td>
`bounding_boxes`
</td>
<td>
A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`
describing the N bounding boxes associated with the image.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
An optional `int`. Defaults to `0`. If `seed` is set to non-zero, the
random number generator is seeded by the given `seed`.  Otherwise, it is
seeded by a random seed.
</td>
</tr><tr>
<td>
`min_object_covered`
</td>
<td>
A Tensor of type `float32`. Defaults to `0.1`. The
cropped area of the image must contain at least this fraction of any
bounding box supplied. The value of this parameter should be non-negative.
In the case of 0, the cropped area does not need to overlap any of the
bounding boxes supplied.
</td>
</tr><tr>
<td>
`aspect_ratio_range`
</td>
<td>
An optional list of `floats`. Defaults to `[0.75,
1.33]`. The cropped area of the image must have an aspect `ratio = width /
height` within this range.
</td>
</tr><tr>
<td>
`area_range`
</td>
<td>
An optional list of `floats`. Defaults to `[0.05, 1]`. The
cropped area of the image must contain a fraction of the supplied image
within this range.
</td>
</tr><tr>
<td>
`max_attempts`
</td>
<td>
An optional `int`. Defaults to `100`. Number of attempts at
generating a cropped region of the image of the specified constraints.
After `max_attempts` failures, return the entire image.
</td>
</tr><tr>
<td>
`use_image_if_no_bounding_boxes`
</td>
<td>
An optional `bool`. Defaults to `False`.
Controls behavior if no bounding boxes supplied. If true, assume an
implicit bounding box covering the whole input. If false, raise an error.
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
A tuple of `Tensor` objects (begin, size, bboxes).
</td>
</tr>
<tr>
<td>
`begin`
</td>
<td>
A `Tensor`. Has the same type as `image_size`. 1-D, containing
`[offset_height, offset_width, 0]`. Provide as input to
  <a href="../../tf/slice.md"><code>tf.slice</code></a>.
</td>
</tr><tr>
<td>
`size`
</td>
<td>
A `Tensor`. Has the same type as `image_size`. 1-D, containing
`[target_height, target_width, -1]`. Provide as input to
  <a href="../../tf/slice.md"><code>tf.slice</code></a>.
</td>
</tr><tr>
<td>
`bboxes`
</td>
<td>
A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing
the distorted bounding box.
Provide as input to <a href="../../tf/image/draw_bounding_boxes.md"><code>tf.image.draw_bounding_boxes</code></a>.
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
If no seed is specified and op determinism is enabled.
</td>
</tr>
</table>

