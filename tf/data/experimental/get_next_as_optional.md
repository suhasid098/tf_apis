description: Returns a <a href="../../../tf/experimental/Optional.md"><code>tf.experimental.Optional</code></a> with the next element of the iterator. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.get_next_as_optional" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.get_next_as_optional

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/ops/iterator_ops.py">View source</a>



Returns a <a href="../../../tf/experimental/Optional.md"><code>tf.experimental.Optional</code></a> with the next element of the iterator. (deprecated)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.get_next_as_optional`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.get_next_as_optional(
    iterator
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/data/Iterator.md#get_next_as_optional"><code>tf.data.Iterator.get_next_as_optional()</code></a> instead.

If the iterator has reached the end of the sequence, the returned
<a href="../../../tf/experimental/Optional.md"><code>tf.experimental.Optional</code></a> will have no value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`iterator`
</td>
<td>
A <a href="../../../tf/data/Iterator.md"><code>tf.data.Iterator</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../tf/experimental/Optional.md"><code>tf.experimental.Optional</code></a> object which either contains the next element
of the iterator (if it exists) or no value.
</td>
</tr>

</table>

