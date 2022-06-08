description: Converts bytearray, bytes, or unicode python input types to bytes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.as_bytes" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.as_bytes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/util/compat.py">View source</a>



Converts `bytearray`, `bytes`, or unicode python input types to `bytes`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.compat.as_bytes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.as_bytes(
    bytes_or_text, encoding=&#x27;utf-8&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Uses utf-8 encoding for text by default.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`bytes_or_text`
</td>
<td>
A `bytearray`, `bytes`, `str`, or `unicode` object.
</td>
</tr><tr>
<td>
`encoding`
</td>
<td>
A string indicating the charset for encoding unicode.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `bytes` object.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If `bytes_or_text` is not a binary or unicode string.
</td>
</tr>
</table>

