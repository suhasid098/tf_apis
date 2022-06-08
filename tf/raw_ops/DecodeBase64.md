description: Decode web-safe base64-encoded strings.
robots: noindex

# tf.raw_ops.DecodeBase64

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Decode web-safe base64-encoded strings.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.DecodeBase64`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.DecodeBase64(
    input, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Input may or may not have padding at the end. See
[EncodeBase64](https://www.tensorflow.org/api_docs/python/tf/io/encode_base64)
for padding. Web-safe means that input must use - and _ instead of + and /.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A `Tensor` of type `string`. Base64 strings to decode.
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
A `Tensor` of type `string`.
</td>
</tr>

</table>

