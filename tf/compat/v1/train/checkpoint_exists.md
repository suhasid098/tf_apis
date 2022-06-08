description: Checks whether a V1 or V2 checkpoint exists with the specified prefix. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.train.checkpoint_exists" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.train.checkpoint_exists

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/checkpoint_management.py">View source</a>



Checks whether a V1 or V2 checkpoint exists with the specified prefix. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.train.checkpoint_exists(
    checkpoint_prefix
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.

This is the recommended way to check if a checkpoint exists, since it takes
into account the naming difference between V1 and V2 formats.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`checkpoint_prefix`
</td>
<td>
the prefix of a V1 or V2 checkpoint, with V2 taking
priority.  Typically the result of `Saver.save()` or that of
<a href="../../../../tf/train/latest_checkpoint.md"><code>tf.train.latest_checkpoint()</code></a>, regardless of sharded/non-sharded or
V1/V2.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A bool, true if a checkpoint referred to by `checkpoint_prefix` exists.
</td>
</tr>

</table>

