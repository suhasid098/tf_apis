description: Perform element-wise concatenation of a list of string tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.strings.join" />
<meta itemprop="path" content="Stable" />
</div>

# tf.strings.join

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/string_ops.py">View source</a>



Perform element-wise concatenation of a list of string tensors.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.string_join`, `tf.compat.v1.strings.join`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.strings.join(
    inputs, separator=&#x27;&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given a list of string tensors of same shape, performs element-wise
concatenation of the strings of the same index in all tensors.


```
>>> tf.strings.join(['abc','def']).numpy()
b'abcdef'
>>> tf.strings.join([['abc','123'],
...                  ['def','456'],
...                  ['ghi','789']]).numpy()
array([b'abcdefghi', b'123456789'], dtype=object)
>>> tf.strings.join([['abc','123'],
...                  ['def','456']],
...                  separator=" ").numpy()
array([b'abc def', b'123 456'], dtype=object)
```

The reduction version of this elementwise operation is
<a href="../../tf/strings/reduce_join.md"><code>tf.strings.reduce_join</code></a>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
A list of <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> objects of same size and <a href="../../tf.md#string"><code>tf.string</code></a> dtype.
</td>
</tr><tr>
<td>
`separator`
</td>
<td>
A string added between each string being joined.
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
A <a href="../../tf.md#string"><code>tf.string</code></a> tensor.
</td>
</tr>

</table>

