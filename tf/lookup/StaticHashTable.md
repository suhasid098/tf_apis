description: A generic hash table that is immutable once initialized.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lookup.StaticHashTable" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="lookup"/>
<meta itemprop="property" content="size"/>
</div>

# tf.lookup.StaticHashTable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>



A generic hash table that is immutable once initialized.

Inherits From: [`TrackableResource`](../../tf/saved_model/experimental/TrackableResource.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lookup.StaticHashTable(
    initializer, default_value, name=None, experimental_is_anonymous=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example usage:



```
>>> keys_tensor = tf.constant(['a', 'b', 'c'])
>>> vals_tensor = tf.constant([7, 8, 9])
>>> input_tensor = tf.constant(['a', 'f'])
>>> table = tf.lookup.StaticHashTable(
...     tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
...     default_value=-1)
>>> table.lookup(input_tensor).numpy()
array([ 7, -1], dtype=int32)
```

Or for more pythonic code:

```
>>> table[input_tensor].numpy()
array([ 7, -1], dtype=int32)
```

The result of a lookup operation has the same shape as the argument:

```
>>> input_tensor = tf.constant([['a', 'b'], ['c', 'd']])
>>> table[input_tensor].numpy()
array([[ 7,  8],
       [ 9, -1]], dtype=int32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`initializer`
</td>
<td>
The table initializer to use. See `HashTable` kernel for
supported key and value types.
</td>
</tr><tr>
<td>
`default_value`
</td>
<td>
The value to use if a key is missing in the table.
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
`experimental_is_anonymous`
</td>
<td>
Whether to use anonymous mode for the
table (default is False). In anonymous mode, the table
resource can only be accessed via a resource handle. It can't
be looked up by a name. When all resource handles pointing to
that resource are gone, the resource will be deleted
automatically.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`default_value`
</td>
<td>
The default value of the table.
</td>
</tr><tr>
<td>
`key_dtype`
</td>
<td>
The table key dtype.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The name of the table.
</td>
</tr><tr>
<td>
`resource_handle`
</td>
<td>
Returns the resource handle associated with this Resource.
</td>
</tr><tr>
<td>
`value_dtype`
</td>
<td>
The table value dtype.
</td>
</tr>
</table>



## Methods

<h3 id="export"><code>export</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export(
    name=None
)
</code></pre>

Returns tensors of all keys and values in the table.


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
A pair of tensors with the first tensor containing all keys and the
second tensors containing all values in the table.
</td>
</tr>

</table>



<h3 id="lookup"><code>lookup</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lookup(
    keys, name=None
)
</code></pre>

Looks up `keys` in a table, outputs the corresponding values.

The `default_value` is used for keys not present in the table.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`keys`
</td>
<td>
Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
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
A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,
otherwise a dense `Tensor`.
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
when `keys` or `default_value` doesn't match the table data
types.
</td>
</tr>
</table>



<h3 id="size"><code>size</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>size(
    name=None
)
</code></pre>

Compute the number of elements in this table.


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
A scalar tensor containing the number of elements in this table.
</td>
</tr>

</table>



<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    keys
)
</code></pre>

Looks up `keys` in a table, outputs the corresponding values.




