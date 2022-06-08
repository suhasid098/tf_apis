description: Creates a single-client mesh.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.create_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.create_mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/mesh_util.py">View source</a>



Creates a single-client mesh.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.create_mesh(
    mesh_dims: Optional[List[Tuple[str, int]]] = None,
    mesh_name: str = &#x27;&#x27;,
    devices: Optional[List[str]] = None,
    device_type: Optional[str] = None
) -> <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

If both `mesh_dims` and `devices` are specified, they must match each otehr.
As a special case, when all arguments are missing, this creates a 1D CPU mesh
with an empty name, assigning all available devices to that dimension.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh_dims`
</td>
<td>
A list of (dim_name, dim_size) tuples. Defaults to a single
batch-parallel dimension called 'x' using all devices. As a special case,
a single-element mesh_dims whose dim_size is -1 also uses all devices.
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
`devices`
</td>
<td>
String representations of devices to use. This is the device part
of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available logical devices.
</td>
</tr><tr>
<td>
`device_type`
</td>
<td>
If `devices` is missing, the type of devices to use. Defaults
to 'CPU'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A single-client mesh created from specified or default arguments.
</td>
</tr>

</table>

