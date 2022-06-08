description: File I/O wrappers without thread locking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.gfile.FastGFile" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="close"/>
<meta itemprop="property" content="flush"/>
<meta itemprop="property" content="next"/>
<meta itemprop="property" content="read"/>
<meta itemprop="property" content="readline"/>
<meta itemprop="property" content="readlines"/>
<meta itemprop="property" content="seek"/>
<meta itemprop="property" content="seekable"/>
<meta itemprop="property" content="size"/>
<meta itemprop="property" content="tell"/>
<meta itemprop="property" content="write"/>
</div>

# tf.compat.v1.gfile.FastGFile

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/platform/gfile.py">View source</a>



File I/O wrappers without thread locking.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.gfile.FastGFile(
    name, mode=&#x27;r&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note, that this  is somewhat like builtin Python  file I/O, but
there are  semantic differences to  make it more  efficient for
some backing filesystems.  For example, a write  mode file will
not  be opened  until the  first  write call  (to minimize  RPC
invocations in network filesystems).



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`mode`
</td>
<td>
Returns the mode in which the file was opened.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Returns the file name.
</td>
</tr>
</table>



## Methods

<h3 id="close"><code>close</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>close()
</code></pre>

Closes the file.

Should be called for the WritableFile to be flushed.

In general, if you use the context manager pattern, you don't need to call
this directly.

```
>>> with tf.io.gfile.GFile("/tmp/x", "w") as f:
...   f.write("asdf\n")
...   f.write("qwer\n")
>>> # implicit f.close() at the end of the block
```

For cloud filesystems, forgetting to call `close()` might result in data
loss as last write might not have been replicated.

<h3 id="flush"><code>flush</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flush()
</code></pre>

Flushes the Writable file.

This only ensures that the data has made its way out of the process without
any guarantees on whether it's written to disk. This means that the
data would survive an application crash but not necessarily an OS crash.

<h3 id="next"><code>next</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next()
</code></pre>




<h3 id="read"><code>read</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>read(
    n=-1
)
</code></pre>

Returns the contents of a file as a string.

Starts reading from current position in file.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`n`
</td>
<td>
Read `n` bytes if `n != -1`. If `n = -1`, reads to end of file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`n` bytes of the file (or whole file) in bytes mode or `n` bytes of the
string if in string (regular) mode.
</td>
</tr>

</table>



<h3 id="readline"><code>readline</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>readline()
</code></pre>

Reads the next line, keeping \n. At EOF, returns ''.


<h3 id="readlines"><code>readlines</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>readlines()
</code></pre>

Returns all lines from the file in a list.


<h3 id="seek"><code>seek</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>seek(
    offset=None, whence=0, position=None
)
</code></pre>

Seeks to the offset in the file. (deprecated arguments)

Deprecated: SOME ARGUMENTS ARE DEPRECATED: `(position)`. They will be removed in a future version.
Instructions for updating:
position is deprecated in favor of the offset argument.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`offset`
</td>
<td>
The byte count relative to the whence argument.
</td>
</tr><tr>
<td>
`whence`
</td>
<td>
Valid values for whence are:
0: start of the file (default)
1: relative to the current position of the file
2: relative to the end of file. `offset` is usually negative.
</td>
</tr>
</table>



<h3 id="seekable"><code>seekable</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>seekable()
</code></pre>

Returns True as FileIO supports random access ops of seek()/tell()


<h3 id="size"><code>size</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>size()
</code></pre>

Returns the size of the file.


<h3 id="tell"><code>tell</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tell()
</code></pre>

Returns the current position in the file.


<h3 id="write"><code>write</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>write(
    file_content
)
</code></pre>

Writes file_content to the file. Appends to the end of the file.


<h3 id="__enter__"><code>__enter__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__enter__()
</code></pre>

Make usable with "with" statement.


<h3 id="__exit__"><code>__exit__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__exit__(
    unused_type, unused_value, unused_traceback
)
</code></pre>

Make usable with "with" statement.


<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>






