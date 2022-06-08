description: Determines whether a path exists or not.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.gfile.exists" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.gfile.exists

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>



Determines whether a path exists or not.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.io.gfile.exists`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.gfile.exists(
    path
)
</code></pre>



<!-- Placeholder for "Used in" -->

```
>>> with open("/tmp/x", "w") as f:
...   f.write("asdf")
...
4
>>> tf.io.gfile.exists("/tmp/x")
True
```

You can also specify the URI scheme for selecting a different filesystem:

```
>>> # for a GCS filesystem path:
>>> # tf.io.gfile.exists("gs://bucket/file")
>>> # for a local filesystem:
>>> with open("/tmp/x", "w") as f:
...   f.write("asdf")
...
4
>>> tf.io.gfile.exists("file:///tmp/x")
True
```

This currently returns `True` for existing directories but don't rely on this
behavior, especially if you are using cloud filesystems (e.g., GCS, S3,
Hadoop):

```
>>> tf.io.gfile.exists("/tmp")
True
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path`
</td>
<td>
string, a path
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if the path exists, whether it's a file or a directory.
False if the path does not exist and there are no filesystem errors.
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
Propagates any errors reported by the FileSystem API.
</td>
</tr>
</table>

