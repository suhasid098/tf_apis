description: Converts all uppercase characters into their respective lowercase replacements.
robots: noindex

# tf.raw_ops.StringLower

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Converts all uppercase characters into their respective lowercase replacements.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.StringLower`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.StringLower(
    input, encoding=&#x27;&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example:



```
>>> tf.strings.lower("CamelCase string and ALL CAPS")
<tf.Tensor: shape=(), dtype=string, numpy=b'camelcase string and all caps'>
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
A `Tensor` of type `string`. The input to be lower-cased.
</td>
</tr><tr>
<td>
`encoding`
</td>
<td>
An optional `string`. Defaults to `""`.
Character encoding of `input`. Allowed values are '' and 'utf-8'.
Value '' is interpreted as ASCII.
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

