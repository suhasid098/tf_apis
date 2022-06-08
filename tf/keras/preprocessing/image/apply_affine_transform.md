description: Applies an affine transformation specified by the parameters given.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.image.apply_affine_transform" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.image.apply_affine_transform

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/preprocessing/image.py#L2212-L2333">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Applies an affine transformation specified by the parameters given.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.preprocessing.image.apply_affine_transform`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.image.apply_affine_transform(
    x,
    theta=0,
    tx=0,
    ty=0,
    shear=0,
    zx=1,
    zy=1,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode=&#x27;nearest&#x27;,
    cval=0.0,
    order=1
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
3D numpy array - a 2D image with one or more channels.
</td>
</tr><tr>
<td>
`theta`
</td>
<td>
Rotation angle in degrees.
</td>
</tr><tr>
<td>
`tx`
</td>
<td>
Width shift.
</td>
</tr><tr>
<td>
`ty`
</td>
<td>
Heigh shift.
</td>
</tr><tr>
<td>
`shear`
</td>
<td>
Shear angle in degrees.
</td>
</tr><tr>
<td>
`zx`
</td>
<td>
Zoom in x direction.
</td>
</tr><tr>
<td>
`zy`
</td>
<td>
Zoom in y direction
</td>
</tr><tr>
<td>
`row_axis`
</td>
<td>
Index of axis for rows (aka Y axis) in the input
image. Direction: left to right.
</td>
</tr><tr>
<td>
`col_axis`
</td>
<td>
Index of axis for columns (aka X axis) in the input
image. Direction: top to bottom.
</td>
</tr><tr>
<td>
`channel_axis`
</td>
<td>
Index of axis for channels in the input image.
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
`order`
</td>
<td>
int, order of interpolation
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The transformed version of the input.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ImportError`
</td>
<td>
if SciPy is not available.
</td>
</tr>
</table>

