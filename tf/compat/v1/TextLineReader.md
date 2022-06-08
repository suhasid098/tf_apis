description: A Reader that outputs the lines of a file delimited by newlines.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.TextLineReader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="num_records_produced"/>
<meta itemprop="property" content="num_work_units_completed"/>
<meta itemprop="property" content="read"/>
<meta itemprop="property" content="read_up_to"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="restore_state"/>
<meta itemprop="property" content="serialize_state"/>
</div>

# tf.compat.v1.TextLineReader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>



A Reader that outputs the lines of a file delimited by newlines.

Inherits From: [`ReaderBase`](../../../tf/compat/v1/ReaderBase.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.TextLineReader(
    skip_header_lines=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Newlines are stripped from the output.
See ReaderBase for supported methods.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`skip_header_lines`
</td>
<td>
An optional int. Defaults to 0.  Number of lines
to skip from the beginning of every file.
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
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`reader_ref`
</td>
<td>
Op that implements the reader.
</td>
</tr><tr>
<td>
`supports_serialize`
</td>
<td>
Whether the Reader implementation can serialize its state.
</td>
</tr>
</table>



## Methods

<h3 id="num_records_produced"><code>num_records_produced</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_records_produced(
    name=None
)
</code></pre>

Returns the number of records this reader has produced.

This is the same as the number of Read executions that have
succeeded.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An int64 Tensor.
</td>
</tr>

</table>



<h3 id="num_work_units_completed"><code>num_work_units_completed</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_work_units_completed(
    name=None
)
</code></pre>

Returns the number of work units this reader has finished processing.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An int64 Tensor.
</td>
</tr>

</table>



<h3 id="read"><code>read</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>read(
    queue, name=None
)
</code></pre>

Returns the next record (key, value) pair produced by a reader.

Will dequeue a work unit from queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has
finished with the previous file).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`queue`
</td>
<td>
A Queue or a mutable string Tensor representing a handle
to a Queue, with string work items.
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of Tensors (key, value).
</td>
</tr>
<tr>
<td>
`key`
</td>
<td>
A string scalar Tensor.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
A string scalar Tensor.
</td>
</tr>
</table>



<h3 id="read_up_to"><code>read_up_to</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>read_up_to(
    queue, num_records, name=None
)
</code></pre>

Returns up to num_records (key, value) pairs produced by a reader.

Will dequeue a work unit from queue if necessary (e.g., when the
Reader needs to start reading from a new file since it has
finished with the previous file).
It may return less than num_records even before the last batch.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`queue`
</td>
<td>
A Queue or a mutable string Tensor representing a handle
to a Queue, with string work items.
</td>
</tr><tr>
<td>
`num_records`
</td>
<td>
Number of records to read.
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of Tensors (keys, values).
</td>
</tr>
<tr>
<td>
`keys`
</td>
<td>
A 1-D string Tensor.
</td>
</tr><tr>
<td>
`values`
</td>
<td>
A 1-D string Tensor.
</td>
</tr>
</table>



<h3 id="reset"><code>reset</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset(
    name=None
)
</code></pre>

Restore a reader to its initial clean state.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The created Operation.
</td>
</tr>

</table>



<h3 id="restore_state"><code>restore_state</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>restore_state(
    state, name=None
)
</code></pre>

Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
Unimplemented error.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`state`
</td>
<td>
A string Tensor.
Result of a SerializeState of a Reader with matching type.
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The created Operation.
</td>
</tr>

</table>



<h3 id="serialize_state"><code>serialize_state</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>serialize_state(
    name=None
)
</code></pre>

Produce a string tensor that encodes the state of a reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A string Tensor.
</td>
</tr>

</table>







 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

Readers are not compatible with eager execution. Instead, please
use <a href="../../../tf/data.md"><code>tf.data</code></a> to get data into your model.


 </devsite-expandable></section>

