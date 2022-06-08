description: Represents a Mesh configuration over a certain list of Mesh Dimensions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.Mesh" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="as_proto"/>
<meta itemprop="property" content="contains_dim"/>
<meta itemprop="property" content="device_type"/>
<meta itemprop="property" content="dim_size"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="from_string"/>
<meta itemprop="property" content="host_mesh"/>
<meta itemprop="property" content="is_remote"/>
<meta itemprop="property" content="local_device_ids"/>
<meta itemprop="property" content="local_device_locations"/>
<meta itemprop="property" content="local_devices"/>
<meta itemprop="property" content="min_global_device_id"/>
<meta itemprop="property" content="num_local_devices"/>
<meta itemprop="property" content="shape"/>
<meta itemprop="property" content="to_string"/>
<meta itemprop="property" content="unravel_index"/>
</div>

# tf.experimental.dtensor.Mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>



Represents a Mesh configuration over a certain list of Mesh Dimensions.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.Mesh(
    dim_names: List[str],
    global_device_ids: np.ndarray,
    local_device_ids: List[int],
    local_devices: List[<a href="../../../tf/compat/v1/DeviceSpec.md"><code>tf.compat.v1.DeviceSpec</code></a>],
    mesh_name: str = &#x27;&#x27;,
    global_devices: Optional[List[tf_device.DeviceSpec]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

A mesh consists of named dimensions with sizes, which describe how a set of
devices are arranged. Defining tensor layouts in terms of mesh dimensions
allows us to efficiently determine the communication required when computing
an operation with tensors of different layouts.

A mesh provides information not only about the placement of the tensors but
also the topology of the underlying devices. For example, we can group 8 TPUs
as a 1-D array for data parallelism or a `2x4` grid for (2-way) data
parallelism and (4-way) model parallelism.

Note: the utilities <a href="../../../tf/experimental/dtensor/create_mesh.md"><code>dtensor.create_mesh</code></a> and
<a href="../../../tf/experimental/dtensor/create_distributed_mesh.md"><code>dtensor.create_distributed_mesh</code></a> provide a simpler API to create meshes for
single- or multi-client use cases.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dim_names`
</td>
<td>
A list of strings indicating dimension names.
</td>
</tr><tr>
<td>
`global_device_ids`
</td>
<td>
An ndarray of global device IDs is used to compose
DeviceSpecs describing the mesh. The shape of this array determines the
size of each mesh dimension. Values in this array should increment
sequentially from 0. This argument is the same for every DTensor client.
</td>
</tr><tr>
<td>
`local_device_ids`
</td>
<td>
A list of local device IDs equal to a subset of values
in global_device_ids. They indicate the position of local devices in the
global mesh. Different DTensor clients must contain distinct
local_device_ids contents. All local_device_ids from all DTensor clients
must cover every element in global_device_ids.
</td>
</tr><tr>
<td>
`local_devices`
</td>
<td>
The list of devices hosted locally. The elements correspond
1:1 to those of local_device_ids.
</td>
</tr><tr>
<td>
`mesh_name`
</td>
<td>
The name of the mesh. Currently, this is rarely used, and is
  mostly used to indicate whether it is a CPU, GPU, or TPU-based mesh.
global_devices (optional): The list of global devices. Set when multiple
  device meshes are in use.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dim_names`
</td>
<td>

</td>
</tr><tr>
<td>
`name`
</td>
<td>

</td>
</tr><tr>
<td>
`size`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="as_proto"><code>as_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_proto() -> layout_pb2.MeshProto
</code></pre>

Returns mesh protobuffer.


<h3 id="contains_dim"><code>contains_dim</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>contains_dim(
    dim_name: str
) -> bool
</code></pre>

Returns True if a Mesh contains the given dimension name.


<h3 id="device_type"><code>device_type</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>device_type() -> str
</code></pre>

Returns the device_type of a Mesh.


<h3 id="dim_size"><code>dim_size</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dim_size(
    dim_name: str
) -> int
</code></pre>

Returns the size of a dimension.


<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_proto(
    proto: layout_pb2.MeshProto
) -> 'Mesh'
</code></pre>

Construct a mesh instance from input `proto`.


<h3 id="from_string"><code>from_string</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_string(
    mesh_str: str
) -> 'Mesh'
</code></pre>

Construct a mesh instance from input `proto`.


<h3 id="host_mesh"><code>host_mesh</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>host_mesh()
</code></pre>

Returns the 1-1 mapped host mesh.


<h3 id="is_remote"><code>is_remote</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_remote() -> bool
</code></pre>

Returns True if a Mesh contains only remote devices.


<h3 id="local_device_ids"><code>local_device_ids</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>local_device_ids() -> List[int]
</code></pre>

Returns a list of local device IDs.


<h3 id="local_device_locations"><code>local_device_locations</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>local_device_locations() -> List[Dict[str, int]]
</code></pre>

Returns a list of local device locations.

A device location is a dictionary from dimension names to indices on those
dimensions.

<h3 id="local_devices"><code>local_devices</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>local_devices() -> List[str]
</code></pre>

Returns a list of local device specs represented as strings.


<h3 id="min_global_device_id"><code>min_global_device_id</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>min_global_device_id() -> int
</code></pre>

Returns the minimum global device ID.


<h3 id="num_local_devices"><code>num_local_devices</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_local_devices() -> int
</code></pre>

Returns the number of local devices.


<h3 id="shape"><code>shape</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>shape() -> List[int]
</code></pre>

Returns the shape of the mesh.


<h3 id="to_string"><code>to_string</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_string() -> str
</code></pre>

Returns string representation of Mesh.


<h3 id="unravel_index"><code>unravel_index</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unravel_index()
</code></pre>

Returns a dictionary from device ID to {dim_name: dim_index}.

For example, for a 3x2 mesh, return this:

```
  { 0: {'x': 0, 'y', 0},
    1: {'x': 0, 'y', 1},
    2: {'x': 1, 'y', 0},
    3: {'x': 1, 'y', 1},
    4: {'x': 2, 'y', 0},
    5: {'x': 2, 'y', 1} }
```

<h3 id="__contains__"><code>__contains__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    dim_name: str
) -> bool
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    dim_name: str
) -> MeshDimension
</code></pre>






