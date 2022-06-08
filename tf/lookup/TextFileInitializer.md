description: Table initializers from a text file.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lookup.TextFileInitializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
</div>

# tf.lookup.TextFileInitializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>



Table initializers from a text file.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.lookup.TextFileInitializer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lookup.TextFileInitializer(
    filename,
    key_dtype,
    key_index,
    value_dtype,
    value_index,
    vocab_size=None,
    delimiter=&#x27;\t&#x27;,
    name=None,
    value_index_offset=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

This initializer assigns one entry in the table for each line in the file.

The key and value type of the table to initialize is given by `key_dtype` and
`value_dtype`.

The key and value content to get from each line is specified by
the `key_index` and `value_index`.

* <a href="../../tf/lookup/TextFileIndex.md#LINE_NUMBER"><code>TextFileIndex.LINE_NUMBER</code></a> means use the line number starting from zero,
  expects data type int64.
* <a href="../../tf/lookup/TextFileIndex.md#WHOLE_LINE"><code>TextFileIndex.WHOLE_LINE</code></a> means use the whole line content, expects data
  type string.
* A value `>=0` means use the index (starting at zero) of the split line based
    on `delimiter`.

For example if we have a file with the following content:

```
>>> import tempfile
>>> f = tempfile.NamedTemporaryFile(delete=False)
>>> content='\n'.join(["emerson 10", "lake 20", "palmer 30",])
>>> f.file.write(content.encode('utf-8'))
>>> f.file.close()
```

The following snippet initializes a table with the first column as keys and
second column as values:

* `emerson -> 10`
* `lake -> 20`
* `palmer -> 30`

```
>>> init= tf.lookup.TextFileInitializer(
...    filename=f.name,
...    key_dtype=tf.string, key_index=0,
...    value_dtype=tf.int64, value_index=1,
...    delimiter=" ")
>>> table = tf.lookup.StaticHashTable(init, default_value=-1)
>>> table.lookup(tf.constant(['palmer','lake','tarkus'])).numpy()
```

Similarly to initialize the whole line as keys and the line number as values.

* `emerson 10 -> 0`
* `lake 20 -> 1`
* `palmer 30 -> 2`

```
>>> init = tf.lookup.TextFileInitializer(
...   filename=f.name,
...   key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
...   value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
>>> table = tf.lookup.StaticHashTable(init, -1)
>>> table.lookup(tf.constant('palmer 30')).numpy()
2
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filename`
</td>
<td>
The filename of the text file to be used for initialization. The
path must be accessible from wherever the graph is initialized (eg.
trainer or eval workers). The filename may be a scalar `Tensor`.
</td>
</tr><tr>
<td>
`key_dtype`
</td>
<td>
The `key` data type.
</td>
</tr><tr>
<td>
`key_index`
</td>
<td>
the index that represents information of a line to get the
table 'key' values from.
</td>
</tr><tr>
<td>
`value_dtype`
</td>
<td>
The `value` data type.
</td>
</tr><tr>
<td>
`value_index`
</td>
<td>
the index that represents information of a line to get the
table 'value' values from.'
</td>
</tr><tr>
<td>
`vocab_size`
</td>
<td>
The number of elements in the file, if known.
</td>
</tr><tr>
<td>
`delimiter`
</td>
<td>
The delimiter to separate fields in a line.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr><tr>
<td>
`value_index_offset`
</td>
<td>
A number to add to all indices extracted from the file
This is useful for cases where a user would like to reserve one or more
low index values for control characters. For instance, if you would
like to ensure that no vocabulary item is mapped to index 0 (so you can
reserve 0 for a masking value), you can set value_index_offset to 1;
this will mean that the first vocabulary element is mapped to 1
instead of 0.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
when the filename is empty, or when the table key and value
data types do not match the expected data types.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`key_dtype`
</td>
<td>
The expected table key dtype.
</td>
</tr><tr>
<td>
`value_dtype`
</td>
<td>
The expected table value dtype.
</td>
</tr>
</table>



## Methods

<h3 id="initialize"><code>initialize</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initialize(
    table
)
</code></pre>

Initializes the table from a text file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`table`
</td>
<td>
The table to be initialized.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The operation that initializes the table.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
when the keys and values data types do not match the table
key and value data types.
</td>
</tr>
</table>





