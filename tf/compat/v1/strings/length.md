description: Computes the length of each string given in the input tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.strings.length" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.strings.length

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/string_ops.py">View source</a>



Computes the length of each string given in the input tensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.strings.length(
    input, name=None, unit=&#x27;BYTE&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

```
>>> strings = tf.constant(['Hello','TensorFlow', '🙂'])
>>> tf.strings.length(strings).numpy() # default counts bytes
array([ 5, 10, 4], dtype=int32)
>>> tf.strings.length(strings, unit="UTF8_CHAR").numpy()
array([ 5, 10, 1], dtype=int32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A `Tensor` of type `string`. The strings for which to compute the
length for each element.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr><tr>
<td>
`unit`
</td>
<td>
An optional `string` from: `"BYTE", "UTF8_CHAR"`. Defaults to
`"BYTE"`. The unit that is counted to compute string length.  One of:
  `"BYTE"` (for the number of bytes in each string) or `"UTF8_CHAR"` (for
  the number of UTF-8 encoded Unicode code points in each string). Results
  are undefined if `unit=UTF8_CHAR` and the `input` strings do not contain
  structurally valid UTF-8.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `int32`, containing the length of the input string in
the same element of the input tensor.
</td>
</tr>

</table>

