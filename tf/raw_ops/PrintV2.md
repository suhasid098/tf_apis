description: Prints a string scalar.
robots: noindex

# tf.raw_ops.PrintV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Prints a string scalar.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.PrintV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.PrintV2(
    input, output_stream=&#x27;stderr&#x27;, end=&#x27;\n&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Prints a string scalar to the desired output_stream.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A `Tensor` of type `string`. The string scalar to print.
</td>
</tr><tr>
<td>
`output_stream`
</td>
<td>
An optional `string`. Defaults to `"stderr"`.
A string specifying the output stream or logging level to print to.
</td>
</tr><tr>
<td>
`end`
</td>
<td>
An optional `string`. Defaults to `"\n"`.
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
The created Operation.
</td>
</tr>

</table>

