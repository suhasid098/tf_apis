description: Recursive directory tree generator for directories.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.gfile.Walk" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.gfile.Walk

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>



Recursive directory tree generator for directories.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.gfile.Walk(
    top, in_order=True
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`top`
</td>
<td>
string, a Directory name
</td>
</tr><tr>
<td>
`in_order`
</td>
<td>
bool, Traverse in order if True, post order if False.  Errors that
happen while listing directories are ignored.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
all its subdirectories and leaf files. That is, each yield looks like:
`(dirname, [subdirname, subdirname, ...], [filename, filename, ...])`.
Each item is a string.
</td>
</tr>

</table>

