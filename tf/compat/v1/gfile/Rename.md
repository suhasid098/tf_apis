description: Rename or move a file / directory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.gfile.Rename" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.gfile.Rename

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>



Rename or move a file / directory.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.gfile.Rename(
    oldname, newname, overwrite=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`oldname`
</td>
<td>
string, pathname for a file
</td>
</tr><tr>
<td>
`newname`
</td>
<td>
string, pathname to which the file needs to be moved
</td>
</tr><tr>
<td>
`overwrite`
</td>
<td>
boolean, if false it's an error for `newname` to be occupied by
an existing file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`errors.OpError`
</td>
<td>
If the operation fails.
</td>
</tr>
</table>

