description: Decode the first frame of a BMP-encoded image to a uint8 tensor.
robots: noindex

# tf.raw_ops.DecodeBmp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Decode the first frame of a BMP-encoded image to a uint8 tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.DecodeBmp`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.DecodeBmp(
    contents, channels=0, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The attr `channels` indicates the desired number of color channels for the
decoded image.

#### Accepted values are:



*   0: Use the number of channels in the BMP-encoded image.
*   3: output an RGB image.
*   4: output an RGBA image.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`contents`
</td>
<td>
A `Tensor` of type `string`. 0-D.  The BMP-encoded image.
</td>
</tr><tr>
<td>
`channels`
</td>
<td>
An optional `int`. Defaults to `0`.
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
A `Tensor` of type `uint8`.
</td>
</tr>

</table>

