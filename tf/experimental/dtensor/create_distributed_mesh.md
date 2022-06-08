description: Creates a single- or multi-client mesh.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.create_distributed_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.create_distributed_mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/mesh_util.py">View source</a>



Creates a single- or multi-client mesh.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.create_distributed_mesh(
    mesh_dims: List[Tuple[str, int]],
    mesh_name: str = &#x27;&#x27;,
    num_global_devices: Optional[int] = None,
    num_clients: Optional[int] = None,
    client_id: Optional[int] = None,
    device_type: str = &#x27;CPU&#x27;
) -> <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

For CPU and GPU meshes, users can choose to use fewer local devices than what
is available. If any argument is missing, it will be extracted from
environment variables. The default values for these environment variables
create a mesh using all devices (common for unit tests).

For TPU meshes, users should not specify any of the nullable arguments. The
DTensor runtime will set these arguments automatically, using all TPU cores
available in the entire cluster.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh_dims`
</td>
<td>
A list of (dim_name, dim_size) tuples.
</td>
</tr><tr>
<td>
`mesh_name`
</td>
<td>
Name of the created mesh. Defaults to ''.
</td>
</tr><tr>
<td>
`num_global_devices`
</td>
<td>
Number of devices in the DTensor cluster. Defaults to
the corresponding environment variable.
</td>
</tr><tr>
<td>
`num_clients`
</td>
<td>
Number of clients in the DTensor cluster. Defaults to the
corresponding environment variable, DTENSOR_NUM_CLIENTS.
</td>
</tr><tr>
<td>
`client_id`
</td>
<td>
This client's ID. Defaults to the corresponding environment
variable, DTENSOR_CLIENT_ID.
</td>
</tr><tr>
<td>
`device_type`
</td>
<td>
Type of device to build the mesh for. Defaults to 'CPU'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A mesh created from specified or default arguments.
</td>
</tr>

</table>

