description: Op is similar to a lightweight Dequeue.
robots: noindex

# tf.raw_ops.Unstage

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Op is similar to a lightweight Dequeue.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.Unstage`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.Unstage(
    dtypes,
    capacity=0,
    memory_limit=0,
    container=&#x27;&#x27;,
    shared_name=&#x27;&#x27;,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The basic functionality is similar to dequeue with many fewer
capabilities and options.  This Op is optimized for performance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dtypes`
</td>
<td>
A list of `tf.DTypes` that has length `>= 1`.
</td>
</tr><tr>
<td>
`capacity`
</td>
<td>
An optional `int` that is `>= 0`. Defaults to `0`.
</td>
</tr><tr>
<td>
`memory_limit`
</td>
<td>
An optional `int` that is `>= 0`. Defaults to `0`.
</td>
</tr><tr>
<td>
`container`
</td>
<td>
An optional `string`. Defaults to `""`.
</td>
</tr><tr>
<td>
`shared_name`
</td>
<td>
An optional `string`. Defaults to `""`.
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
A list of `Tensor` objects of type `dtypes`.
</td>
</tr>

</table>

