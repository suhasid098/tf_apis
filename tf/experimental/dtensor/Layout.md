description: Represents the layout information of a DTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.Layout" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="as_proto"/>
<meta itemprop="property" content="batch_sharded"/>
<meta itemprop="property" content="delete"/>
<meta itemprop="property" content="from_str"/>
<meta itemprop="property" content="from_string"/>
<meta itemprop="property" content="inner_sharded"/>
<meta itemprop="property" content="is_fully_replicated"/>
<meta itemprop="property" content="mesh_proto"/>
<meta itemprop="property" content="num_shards"/>
<meta itemprop="property" content="offset_to_shard"/>
<meta itemprop="property" content="offset_tuple_to_global_index"/>
<meta itemprop="property" content="replicated"/>
<meta itemprop="property" content="serialized_string"/>
<meta itemprop="property" content="to_string"/>
<meta itemprop="property" content="unravel"/>
</div>

# tf.experimental.dtensor.Layout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>



Represents the layout information of a DTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.Layout(
    sharding_specs: List[str],
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

A layout describes how a distributed tensor is partitioned across a mesh (and
thus across devices). For each axis of the tensor, the corresponding
sharding spec indicates which dimension of the mesh it is sharded over. A
special sharding spec `UNSHARDED` indicates that axis is replicated on
all the devices of that mesh.

For example, let's consider a 1-D mesh:

```
Mesh(["TPU:0", "TPU:1", "TPU:2", "TPU:3", "TPU:4", "TPU:5"], [("x", 6)])
```

This mesh arranges 6 TPU devices into a 1-D array. `Layout([UNSHARDED], mesh)`
is a layout for rank-1 tensor which is replicated on the 6 devices.

For another example, let's consider a 2-D mesh:

```
Mesh(["TPU:0", "TPU:1", "TPU:2", "TPU:3", "TPU:4", "TPU:5"],
     [("x", 3), ("y", 2)])
```

This mesh arranges 6 TPU devices into a `3x2` 2-D array.
`Layout(["x", UNSHARDED], mesh)` is a layout for rank-2 tensor whose first
axis is sharded on mesh dimension "x" and the second axis is replicated. If we
place `np.arange(6).reshape((3, 2))` using this layout, the individual
components tensors would look like:

```
Device  |  Component
 TPU:0     [[0, 1]]
 TPU:1     [[0, 1]]
 TPU:2     [[2, 3]]
 TPU:3     [[2, 3]]
 TPU:4     [[4, 5]]
 TPU:5     [[4, 5]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sharding_specs`
</td>
<td>
List of sharding specifications, each corresponding to a
tensor axis. Each specification (dim_sharding) can either be a mesh
dimension or the special value UNSHARDED.
</td>
</tr><tr>
<td>
`mesh`
</td>
<td>
A mesh configuration for the Tensor.
</td>
</tr>
</table>



## Methods

<h3 id="as_proto"><code>as_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_proto() -> layout_pb2.LayoutProto
</code></pre>

Create a proto representation of a layout.


<h3 id="batch_sharded"><code>batch_sharded</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>batch_sharded(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    batch_dim: str,
    rank: int
) -> 'Layout'
</code></pre>

Returns a layout sharded on batch dimension.


<h3 id="delete"><code>delete</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete(
    dims: List[int]
) -> 'Layout'
</code></pre>

Returns the layout with the give dimensions deleted.


<h3 id="from_str"><code>from_str</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_str(
    layout_str: bytes
) -> 'Layout'
</code></pre>

Creates an instance from a serialized Protobuf binary string.


<h3 id="from_string"><code>from_string</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_string(
    layout_str: str
) -> 'Layout'
</code></pre>

Creates an instance from a human-readable string.


<h3 id="inner_sharded"><code>inner_sharded</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>inner_sharded(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    inner_dim: str,
    rank: int
) -> 'Layout'
</code></pre>

Returns a layout sharded on inner dimension.


<h3 id="is_fully_replicated"><code>is_fully_replicated</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_fully_replicated() -> bool
</code></pre>

Returns True if all tensor axes are replicated.


<h3 id="mesh_proto"><code>mesh_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mesh_proto() -> layout_pb2.MeshProto
</code></pre>

Returns the underlying mesh in Protobuf format.


<h3 id="num_shards"><code>num_shards</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_shards(
    idx: int
) -> int
</code></pre>

Returns the number of shards for tensor dimension `idx`.


<h3 id="offset_to_shard"><code>offset_to_shard</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>offset_to_shard()
</code></pre>

Mapping from offset in a flattened list to shard index.


<h3 id="offset_tuple_to_global_index"><code>offset_tuple_to_global_index</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>offset_tuple_to_global_index(
    offset_tuple
)
</code></pre>

Mapping from offset to index in global tensor.


<h3 id="replicated"><code>replicated</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>replicated(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    rank: int
) -> 'Layout'
</code></pre>

Returns a replicated layout of rank `rank`.


<h3 id="serialized_string"><code>serialized_string</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>serialized_string() -> bytes
</code></pre>

Returns a serialized Protobuf binary string representation.


<h3 id="to_string"><code>to_string</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_string() -> str
</code></pre>

Returns a human-readable string representation.


<h3 id="unravel"><code>unravel</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unravel(
    unpacked_tensors: List[np.ndarray]
) -> np.ndarray
</code></pre>

Convert a flattened list of shards into a sharded array.


<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.




