description: Shards computation along the batch dimension for parallel execution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.tpu.batch_parallel" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.tpu.batch_parallel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu.py">View source</a>



Shards `computation` along the batch dimension for parallel execution.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.tpu.batch_parallel(
    computation: Callable[..., Any],
    inputs: Optional[List[List[Optional[core_types.Tensor]]]] = None,
    num_shards: int = 1,
    infeed_queue: Optional[tpu_feed.InfeedQueue] = None,
    device_assignment: Optional[<a href="../../../../tf/tpu/experimental/DeviceAssignment.md"><code>tf.tpu.experimental.DeviceAssignment</code></a>] = None,
    name: Optional[Text] = None,
    xla_options: Optional[<a href="../../../../tf/tpu/XLAOptions.md"><code>tf.tpu.XLAOptions</code></a>] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Convenience wrapper around shard().

`inputs` must be a list of Tensors or None (equivalent to an empty list).
Each input is split into `num_shards` pieces along the 0-th dimension, and
computation is applied to each shard in parallel.

Tensors are broadcast to all shards if they are lexically captured by
`computation`. e.g.,

x = tf.constant(7)
def computation():
  return x + 3
... = shard(computation, ...)

The outputs from all shards are concatenated back together along their 0-th
dimension.

Inputs and outputs of the computation must be at least rank-1 Tensors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`computation`
</td>
<td>
A Python function that builds a computation to apply to each
shard of the input.
</td>
</tr><tr>
<td>
`inputs`
</td>
<td>
A list of input tensors or None (equivalent to an empty list). The
0-th dimension of each Tensor must have size divisible by `num_shards`.
</td>
</tr><tr>
<td>
`num_shards`
</td>
<td>
The number of shards.
</td>
</tr><tr>
<td>
`infeed_queue`
</td>
<td>
If not `None`, the `InfeedQueue` from which to append a tuple
of arguments as inputs to `computation`.
</td>
</tr><tr>
<td>
`device_assignment`
</td>
<td>
If not `None`, a `DeviceAssignment` describing the
mapping between logical cores in the computation with physical cores in
the TPU topology. Uses a default device assignment if `None`. The
`DeviceAssignment` may be omitted if each shard of the computation uses
only one core, and there is either only one shard, or the number of shards
is equal to the number of cores in the TPU system.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(Deprecated) Does nothing.
</td>
</tr><tr>
<td>
`xla_options`
</td>
<td>
An instance of `tpu.XLAOptions` which indicates the options
passed to XLA compiler. Use `None` for default options.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of output tensors.
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
If `num_shards <= 0`
</td>
</tr>
</table>

