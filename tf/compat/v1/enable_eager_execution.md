description: Enables eager execution for the lifetime of this program.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.enable_eager_execution" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.enable_eager_execution

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Enables eager execution for the lifetime of this program.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This function is not necessary if you are using TF2. Eager execution is
enabled by default.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Eager execution provides an imperative interface to TensorFlow. With eager
execution enabled, TensorFlow functions execute operations immediately (as
opposed to adding to a graph to be executed later in a <a href="../../../tf/compat/v1/Session.md"><code>tf.compat.v1.Session</code></a>)
and
return concrete values (as opposed to symbolic references to a node in a
computational graph).

#### For example:



```python
tf.compat.v1.enable_eager_execution()

# After eager execution is enabled, operations are executed as they are
# defined and Tensor objects hold concrete values, which can be accessed as
# numpy.ndarray`s through the numpy() method.
assert tf.multiply(6, 7).numpy() == 42
```

Eager execution cannot be enabled after TensorFlow APIs have been used to
create or execute graphs. It is typically recommended to invoke this function
at program startup and not in a library (as most libraries should be usable
both with and without eager execution).



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`config`
</td>
<td>
(Optional.) A <a href="../../../tf/compat/v1/ConfigProto.md"><code>tf.compat.v1.ConfigProto</code></a> to use to configure the
environment in which operations are executed. Note that
<a href="../../../tf/compat/v1/ConfigProto.md"><code>tf.compat.v1.ConfigProto</code></a> is also used to configure graph execution (via
<a href="../../../tf/compat/v1/Session.md"><code>tf.compat.v1.Session</code></a>) and many options within <a href="../../../tf/compat/v1/ConfigProto.md"><code>tf.compat.v1.ConfigProto</code></a>
are not implemented (or are irrelevant) when eager execution is enabled.
</td>
</tr><tr>
<td>
`device_policy`
</td>
<td>
(Optional.) Policy controlling how operations requiring
inputs on a specific device (e.g., a GPU 0) handle inputs on a different
device  (e.g. GPU 1 or CPU). When set to None, an appropriate value will
be picked automatically. The value picked may change between TensorFlow
releases.
Valid values:
- tf.contrib.eager.DEVICE_PLACEMENT_EXPLICIT: raises an error if the
  placement is not correct.
- tf.contrib.eager.DEVICE_PLACEMENT_WARN: copies the tensors which are not
  on the right device but logs a warning.
- tf.contrib.eager.DEVICE_PLACEMENT_SILENT: silently copies the tensors.
  Note that this may hide performance problems as there is no notification
  provided when operations are blocked on the tensor being copied between
  devices.
- tf.contrib.eager.DEVICE_PLACEMENT_SILENT_FOR_INT32: silently copies
  int32 tensors, raising errors on the other ones.
</td>
</tr><tr>
<td>
`execution_mode`
</td>
<td>
(Optional.) Policy controlling how operations dispatched are
actually executed. When set to None, an appropriate value will be picked
automatically. The value picked may change between TensorFlow releases.
Valid values:
- tf.contrib.eager.SYNC: executes each operation synchronously.
- tf.contrib.eager.ASYNC: executes each operation asynchronously. These
  operations may return "non-ready" handles.
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
If eager execution is enabled after creating/executing a
TensorFlow graph, or if options provided conflict with a previous call
to this function.
</td>
</tr>
</table>

