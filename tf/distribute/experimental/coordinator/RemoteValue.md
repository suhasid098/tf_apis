description: An asynchronously available value of a scheduled function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.experimental.coordinator.RemoteValue" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="fetch"/>
<meta itemprop="property" content="get"/>
</div>

# tf.distribute.experimental.coordinator.RemoteValue

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/coordinator/values.py">View source</a>



An asynchronously available value of a scheduled function.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.distribute.coordinator.RemoteValue`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

This class is used as the return value of
<a href="../../../../tf/distribute/experimental/coordinator/ClusterCoordinator.md#schedule"><code>tf.distribute.experimental.coordinator.ClusterCoordinator.schedule</code></a> where
the underlying value becomes available at a later time once the function has
been executed.

Using <a href="../../../../tf/distribute/experimental/coordinator/RemoteValue.md"><code>tf.distribute.experimental.coordinator.RemoteValue</code></a> as an input to
a subsequent function scheduled with
<a href="../../../../tf/distribute/experimental/coordinator/ClusterCoordinator.md#schedule"><code>tf.distribute.experimental.coordinator.ClusterCoordinator.schedule</code></a> is
currently not supported.

#### Example:



```python
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver=...)
coordinator = (
    tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))

with strategy.scope():
  v1 = tf.Variable(initial_value=0.0)
  v2 = tf.Variable(initial_value=1.0)

@tf.function
def worker_fn():
  v1.assign_add(0.1)
  v2.assign_sub(0.2)
  return v1.read_value() / v2.read_value()

result = coordinator.schedule(worker_fn)
# Note that `fetch()` gives the actual result instead of a `tf.Tensor`.
assert result.fetch() == 0.125

for _ in range(10):
  # `worker_fn` will be run on arbitrary workers that are available. The
  # `result` value will be available later.
  result = coordinator.schedule(worker_fn)
```

## Methods

<h3 id="fetch"><code>fetch</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/coordinator/values.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fetch()
</code></pre>

Wait for the result of `RemoteValue` and return the numpy result.

This makes the value concrete by copying the remote value to local.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The numpy array structure of the actual output of the <a href="../../../../tf/function.md"><code>tf.function</code></a>
associated with this `RemoteValue`, previously returned by a
<a href="../../../../tf/distribute/experimental/coordinator/ClusterCoordinator.md#schedule"><code>tf.distribute.experimental.coordinator.ClusterCoordinator.schedule</code></a> call.
This can be a single value, or a structure of values, depending on the
output of the <a href="../../../../tf/function.md"><code>tf.function</code></a>.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`tf.errors.CancelledError`
</td>
<td>
If the function that produces this `RemoteValue`
is aborted or cancelled due to failure.
</td>
</tr>
</table>



<h3 id="get"><code>get</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/coordinator/values.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get()
</code></pre>

Wait for the result of `RemoteValue` and return the tensor result.

This makes the value concrete by copying the remote tensor to local.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The actual output (in the form of <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a>s) of the <a href="../../../../tf/function.md"><code>tf.function</code></a>
associated with this `RemoteValue`, previously returned by a
<a href="../../../../tf/distribute/experimental/coordinator/ClusterCoordinator.md#schedule"><code>tf.distribute.experimental.coordinator.ClusterCoordinator.schedule</code></a> call.
This can be a single Tensor, or a structure of Tensors, depending on the
output of the <a href="../../../../tf/function.md"><code>tf.function</code></a>.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`tf.errors.CancelledError`
</td>
<td>
If the function that produces this `RemoteValue`
is aborted or cancelled due to failure.
</td>
</tr>
</table>





