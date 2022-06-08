description: Performs a random rotation of a Numpy image tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.image.random_rotation" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.image.random_rotation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/preprocessing/image.py#L1922-L1962">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Performs a random rotation of a Numpy image tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.preprocessing.image.random_rotation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.image.random_rotation(
    x,
    rg,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode=&#x27;nearest&#x27;,
    cval=0.0,
    interpolation_order=1
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/image/random_rotation.md"><code>tf.keras.preprocessing.image.random_rotation</code></a> does not operate on
tensors and is not recommended for new code. Prefer
<a href="../../../../tf/keras/layers/RandomRotation.md"><code>tf.keras.layers.RandomRotation</code></a> which provides equivalent functionality as a
preprocessing layer. For more information, see the tutorial for
[augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
the [preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
Input tensor. Must be 3D.
</td>
</tr><tr>
<td>
`rg`
</td>
<td>
Rotation range, in degrees.
</td>
</tr><tr>
<td>
`row_axis`
</td>
<td>
Index of axis for rows in the input tensor.
</td>
</tr><tr>
<td>
`col_axis`
</td>
<td>
Index of axis for columns in the input tensor.
</td>
</tr><tr>
<td>
`channel_axis`
</td>
<td>
Index of axis for channels in the input tensor.
</td>
</tr><tr>
<td>
`fill_mode`
</td>
<td>
Points outside the boundaries of the input
are filled according to the given mode
(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
</td>
</tr><tr>
<td>
`cval`
</td>
<td>
Value used for points outside the boundaries
of the input if `mode='constant'`.
</td>
</tr><tr>
<td>
`interpolation_order`
</td>
<td>
int, order of spline interpolation.
see `ndimage.interpolation.affine_transform`
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Rotated Numpy image tensor.
</td>
</tr>

</table>

