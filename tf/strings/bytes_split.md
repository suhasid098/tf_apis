description: Split string elements of input into bytes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.strings.bytes_split" />
<meta itemprop="path" content="Stable" />
</div>

# tf.strings.bytes_split

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/ragged_string_ops.py">View source</a>



Split string elements of `input` into bytes.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.strings.bytes_split`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.strings.bytes_split(
    input, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Examples:



```
>>> tf.strings.bytes_split('hello').numpy()
array([b'h', b'e', b'l', b'l', b'o'], dtype=object)
>>> tf.strings.bytes_split(['hello', '123'])
<tf.RaggedTensor [[b'h', b'e', b'l', b'l', b'o'], [b'1', b'2', b'3']]>
```

Note that this op splits strings into bytes, not unicode characters.  To
split strings into unicode characters, use <a href="../../tf/strings/unicode_split.md"><code>tf.strings.unicode_split</code></a>.

See also: <a href="../../tf/io/decode_raw.md"><code>tf.io.decode_raw</code></a>, <a href="../../tf/strings/split.md"><code>tf.strings.split</code></a>, <a href="../../tf/strings/unicode_split.md"><code>tf.strings.unicode_split</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A string `Tensor` or `RaggedTensor`: the strings to split.  Must
have a statically known rank (`N`).
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
A `RaggedTensor` of rank `N+1`: the bytes that make up the source strings.
</td>
</tr>

</table>

