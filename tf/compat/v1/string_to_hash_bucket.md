description: Converts each string in the input Tensor to its hash mod by a number of buckets.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.string_to_hash_bucket" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.string_to_hash_bucket

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/string_ops.py">View source</a>



Converts each string in the input Tensor to its hash mod by a number of buckets.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.strings.to_hash_bucket`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.string_to_hash_bucket(
    string_tensor=None, num_buckets=None, name=None, input=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time.
This functionality will be deprecated and it's recommended to use
`tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`string_tensor`
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`num_buckets`
</td>
<td>
An `int` that is `>= 1`. The number of buckets.
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
A `Tensor` of type `int64`.
</td>
</tr>

</table>

