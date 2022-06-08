description: Additional APIs for algorithms that need to be distribution-aware.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.distribute.StrategyExtended" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_reduce_to"/>
<meta itemprop="property" content="broadcast_to"/>
<meta itemprop="property" content="call_for_each_replica"/>
<meta itemprop="property" content="colocate_vars_with"/>
<meta itemprop="property" content="experimental_make_numpy_dataset"/>
<meta itemprop="property" content="experimental_run_steps_on_iterator"/>
<meta itemprop="property" content="non_slot_devices"/>
<meta itemprop="property" content="read_var"/>
<meta itemprop="property" content="reduce_to"/>
<meta itemprop="property" content="update"/>
<meta itemprop="property" content="update_non_slot"/>
<meta itemprop="property" content="value_container"/>
<meta itemprop="property" content="variable_created_in_scope"/>
</div>

# tf.compat.v1.distribute.StrategyExtended

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>



Additional APIs for algorithms that need to be distribution-aware.

Inherits From: [`StrategyExtended`](../../../../tf/distribute/StrategyExtended.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.distribute.StrategyExtended(
    container_strategy
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: For most usage of <a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>, there should be no need to
call these methods, since TensorFlow libraries (such as optimizers) already
call these methods when needed on your behalf.


Some common use cases of functions on this page:

* _Locality_

<a href="../../../../tf/distribute/DistributedValues.md"><code>tf.distribute.DistributedValues</code></a> can have the same _locality_ as a
_distributed variable_, which leads to a mirrored value residing on the same
devices as the variable (as opposed to the compute devices). Such values may
be passed to a call to <a href="../../../../tf/distribute/StrategyExtended.md#update"><code>tf.distribute.StrategyExtended.update</code></a> to update the
value of a variable. You may use
<a href="../../../../tf/distribute/StrategyExtended.md#colocate_vars_with"><code>tf.distribute.StrategyExtended.colocate_vars_with</code></a> to give a variable the
same locality as another variable. You may convert a "PerReplica" value to a
variable's locality by using <a href="../../../../tf/distribute/StrategyExtended.md#reduce_to"><code>tf.distribute.StrategyExtended.reduce_to</code></a> or
<a href="../../../../tf/distribute/StrategyExtended.md#batch_reduce_to"><code>tf.distribute.StrategyExtended.batch_reduce_to</code></a>.

* _How to update a distributed variable_

A distributed variable is variables created on multiple devices. As discussed
in the [glossary](https://www.tensorflow.org/api_docs/python/tf/distribute),
mirrored variable and SyncOnRead variable are two examples. The standard
pattern for updating distributed variables is to:

1. In your function passed to <a href="../../../../tf/distribute/Strategy.md#run"><code>tf.distribute.Strategy.run</code></a>,
   compute a list of (update, variable) pairs. For example, the update might
   be a gradient of the loss with respect to the variable.
2. Switch to cross-replica mode by calling
   `tf.distribute.get_replica_context().merge_call()` with the updates and
   variables as arguments.
3. Call
   <a href="../../../../tf/distribute/StrategyExtended.md#reduce_to"><code>tf.distribute.StrategyExtended.reduce_to(VariableAggregation.SUM, t, v)</code></a>
   (for one variable) or <a href="../../../../tf/distribute/StrategyExtended.md#batch_reduce_to"><code>tf.distribute.StrategyExtended.batch_reduce_to</code></a>
   (for a list of variables) to sum the updates.
4. Call <a href="../../../../tf/distribute/StrategyExtended.md#update"><code>tf.distribute.StrategyExtended.update(v)</code></a> for each variable to update
   its value.

Steps 2 through 4 are done automatically by class
<a href="../../../../tf/keras/optimizers/Optimizer.md"><code>tf.keras.optimizers.Optimizer</code></a> if you call its
<a href="../../../../tf/keras/optimizers/Optimizer.md#apply_gradients"><code>tf.keras.optimizers.Optimizer.apply_gradients</code></a> method in a replica context.

In fact, a higher-level solution to update a distributed variable is by
calling `assign` on the variable as you would do to a regular <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a>.
You can call the method in both _replica context_ and _cross-replica context_.
For a _mirrored variable_, calling `assign` in _replica context_ requires you
to specify the `aggregation` type in the variable constructor. In that case,
the context switching and sync described in steps 2 through 4 are handled for
you. If you call `assign` on _mirrored variable_ in _cross-replica context_,
you can only assign a single value or assign values from another mirrored
variable or a mirrored <a href="../../../../tf/distribute/DistributedValues.md"><code>tf.distribute.DistributedValues</code></a>. For a _SyncOnRead
variable_, in _replica context_, you can simply call `assign` on it and no
aggregation happens under the hood. In _cross-replica context_, you can only
assign a single value to a SyncOnRead variable. One example case is restoring
from a checkpoint: if the `aggregation` type of the variable is
<a href="../../../../tf/VariableAggregation.md#SUM"><code>tf.VariableAggregation.SUM</code></a>, it is assumed that replica values were added
before checkpointing, so at the time of restoring, the value is divided by
the number of replicas and then assigned to each replica; if the `aggregation`
type is <a href="../../../../tf/VariableAggregation.md#MEAN"><code>tf.VariableAggregation.MEAN</code></a>, the value is assigned to each replica
directly.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`experimental_between_graph`
</td>
<td>
Whether the strategy uses between-graph replication or not.

This is expected to return a constant value that will not be changed
throughout its life cycle.
</td>
</tr><tr>
<td>
`experimental_require_static_shapes`
</td>
<td>
Returns `True` if static shape is required; `False` otherwise.
</td>
</tr><tr>
<td>
`experimental_should_init`
</td>
<td>
Whether initialization is needed.
</td>
</tr><tr>
<td>
`parameter_devices`
</td>
<td>
Returns the tuple of all devices used to place variables.
</td>
</tr><tr>
<td>
`should_checkpoint`
</td>
<td>
Whether checkpointing is needed.
</td>
</tr><tr>
<td>
`should_save_summary`
</td>
<td>
Whether saving summaries is needed.
</td>
</tr><tr>
<td>
`worker_devices`
</td>
<td>
Returns the tuple of all devices used to for compute replica execution.

</td>
</tr>
</table>



## Methods

<h3 id="batch_reduce_to"><code>batch_reduce_to</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_reduce_to(
    reduce_op, value_destination_pairs, options=None
)
</code></pre>

Combine multiple `reduce_to` calls into one for faster execution.

Similar to `reduce_to`, but accepts a list of (value, destinations) pairs.
It's more efficient than reduce each value separately.

This API currently can only be called in cross-replica context. Other
variants to reduce values across replicas are:
* <a href="../../../../tf/distribute/StrategyExtended.md#reduce_to"><code>tf.distribute.StrategyExtended.reduce_to</code></a>: the non-batch version of
  this API.
* <a href="../../../../tf/distribute/ReplicaContext.md#all_reduce"><code>tf.distribute.ReplicaContext.all_reduce</code></a>: the counterpart of this API
  in replica context. It supports both batched and non-batched all-reduce.
* <a href="../../../../tf/distribute/Strategy.md#reduce"><code>tf.distribute.Strategy.reduce</code></a>: a more convenient method to reduce
  to the host in cross-replica context.

See `reduce_to` for more information.

```
>>> @tf.function
... def step_fn(var):
...
...   def merge_fn(strategy, value, var):
...     # All-reduce the value. Note that `value` here is a
...     # `tf.distribute.DistributedValues`.
...     reduced = strategy.extended.batch_reduce_to(
...         tf.distribute.ReduceOp.SUM, [(value, var)])[0]
...     strategy.extended.update(var, lambda var, value: var.assign(value),
...         args=(reduced,))
...
...   value = tf.identity(1.)
...   tf.distribute.get_replica_context().merge_call(merge_fn,
...     args=(value, var))
>>>
>>> def run(strategy):
...   with strategy.scope():
...     v = tf.Variable(0.)
...     strategy.run(step_fn, args=(v,))
...     return v
>>>
>>> run(tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"]))
MirroredVariable:{
  0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>,
  1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=2.0>
}
>>> run(tf.distribute.experimental.CentralStorageStrategy(
...     compute_devices=["GPU:0", "GPU:1"], parameter_device="CPU:0"))
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
>>> run(tf.distribute.OneDeviceStrategy("GPU:0"))
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`reduce_op`
</td>
<td>
a <a href="../../../../tf/distribute/ReduceOp.md"><code>tf.distribute.ReduceOp</code></a> value specifying how values should
be combined. Allows using string representation of the enum such as
"SUM", "MEAN".
</td>
</tr><tr>
<td>
`value_destination_pairs`
</td>
<td>
a sequence of (value, destinations) pairs. See
`tf.distribute.Strategy.reduce_to` for descriptions.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
a <a href="../../../../tf/distribute/experimental/CommunicationOptions.md"><code>tf.distribute.experimental.CommunicationOptions</code></a>. Options to
perform collective operations. This overrides the default options if the
<a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> takes one in the constructor. See
<a href="../../../../tf/distribute/experimental/CommunicationOptions.md"><code>tf.distribute.experimental.CommunicationOptions</code></a> for details of the
options.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of reduced values, one per pair in `value_destination_pairs`.
</td>
</tr>

</table>



<h3 id="broadcast_to"><code>broadcast_to</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>broadcast_to(
    tensor, destinations
)
</code></pre>

Mirror a tensor on one device to all worker devices.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensor`
</td>
<td>
A Tensor value to broadcast.
</td>
</tr><tr>
<td>
`destinations`
</td>
<td>
A mirrored variable or device string specifying the
destination devices to copy `tensor` to.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A value mirrored to `destinations` devices.
</td>
</tr>

</table>



<h3 id="call_for_each_replica"><code>call_for_each_replica</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call_for_each_replica(
    fn, args=(), kwargs=None
)
</code></pre>

Run `fn` once per replica.

`fn` may call `tf.get_replica_context()` to access methods such as
`replica_id_in_sync_group` and `merge_call()`.

`merge_call()` is used to communicate between the replicas and
re-enter the cross-replica context. All replicas pause their execution
having encountered a `merge_call()` call. After that the
`merge_fn`-function is executed. Its results are then unwrapped and
given back to each replica call. After that execution resumes until
`fn` is complete or encounters another `merge_call()`.  Example:

```python
# Called once in "cross-replica" context.
def merge_fn(distribution, three_plus_replica_id):
  # sum the values across replicas
  return sum(distribution.experimental_local_results(three_plus_replica_id))

# Called once per replica in `distribution`, in a "replica" context.
def fn(three):
  replica_ctx = tf.get_replica_context()
  v = three + replica_ctx.replica_id_in_sync_group
  # Computes the sum of the `v` values across all replicas.
  s = replica_ctx.merge_call(merge_fn, args=(v,))
  return s + v

with distribution.scope():
  # in "cross-replica" context
  ...
  merged_results = distribution.run(fn, args=[3])
  # merged_results has the values from every replica execution of `fn`.
  # This statement prints a list:
  print(distribution.experimental_local_results(merged_results))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fn`
</td>
<td>
function to run (will be run once per replica).
</td>
</tr><tr>
<td>
`args`
</td>
<td>
Tuple or list with positional arguments for `fn`.
</td>
</tr><tr>
<td>
`kwargs`
</td>
<td>
Dict with keyword arguments for `fn`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Merged return value of `fn` across all replicas.
</td>
</tr>

</table>



<h3 id="colocate_vars_with"><code>colocate_vars_with</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>colocate_vars_with(
    colocate_with_variable
)
</code></pre>

Scope that controls which devices variables will be created on.

No operations should be added to the graph inside this scope, it
should only be used when creating variables (some implementations
work by changing variable creation, others work by using a
tf.compat.v1.colocate_with() scope).

This may only be used inside `self.scope()`.

#### Example usage:



```
with strategy.scope():
  var1 = tf.Variable(...)
  with strategy.extended.colocate_vars_with(var1):
    # var2 and var3 will be created on the same device(s) as var1
    var2 = tf.Variable(...)
    var3 = tf.Variable(...)

  def fn(v1, v2, v3):
    # operates on v1 from var1, v2 from var2, and v3 from var3

  # `fn` runs on every device `var1` is on, `var2` and `var3` will be there
  # too.
  strategy.extended.update(var1, fn, args=(var2, var3))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`colocate_with_variable`
</td>
<td>
A variable created in this strategy's `scope()`.
Variables created while in the returned context manager will be on the
same set of devices as `colocate_with_variable`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A context manager.
</td>
</tr>

</table>



<h3 id="experimental_make_numpy_dataset"><code>experimental_make_numpy_dataset</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_make_numpy_dataset(
    numpy_input, session=None
)
</code></pre>

Makes a dataset for input provided via a numpy array.

This avoids adding `numpy_input` as a large constant in the graph,
and copies the data to the machine or machines that will be processing
the input.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`numpy_input`
</td>
<td>
A nest of NumPy input arrays that will be distributed evenly
across all replicas. Note that lists of Numpy arrays are stacked, as
that is normal <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> behavior.
</td>
</tr><tr>
<td>
`session`
</td>
<td>
(TensorFlow v1.x graph execution only) A session used for
initialization.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> representing `numpy_input`.
</td>
</tr>

</table>



<h3 id="experimental_run_steps_on_iterator"><code>experimental_run_steps_on_iterator</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_run_steps_on_iterator(
    fn, iterator, iterations=1, initial_loop_values=None
)
</code></pre>

DEPRECATED: please use `run` instead. (deprecated)

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
please use `run` instead.

Run `fn` with input from `iterator` for `iterations` times.

This method can be used to run a step function for training a number of
times using input from a dataset.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fn`
</td>
<td>
function to run using this distribution strategy. The function must
have the following signature: `def fn(context, inputs)`. `context` is an
  instance of `MultiStepContext` that will be passed when `fn` is run.
  `context` can be used to specify the outputs to be returned from `fn`
  by calling `context.set_last_step_output`. It can also be used to
  capture non tensor outputs by `context.set_non_tensor_output`. See
  `MultiStepContext` documentation for more information. `inputs` will
  have same type/structure as `iterator.get_next()`. Typically, `fn`
  will use `call_for_each_replica` method of the strategy to distribute
  the computation over multiple replicas.
</td>
</tr><tr>
<td>
`iterator`
</td>
<td>
Iterator of a dataset that represents the input for `fn`. The
caller is responsible for initializing the iterator as needed.
</td>
</tr><tr>
<td>
`iterations`
</td>
<td>
(Optional) Number of iterations that `fn` should be run.
Defaults to 1.
</td>
</tr><tr>
<td>
`initial_loop_values`
</td>
<td>
(Optional) Initial values to be passed into the
loop that runs `fn`. Defaults to `None`. 
  initial_loop_values argument when we have a mechanism to infer the
  outputs of `fn`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Returns the `MultiStepContext` object which has the following properties,
among other things:
  - run_op: An op that runs `fn` `iterations` times.
  - last_step_outputs: A dictionary containing tensors set using
  `context.set_last_step_output`. Evaluating this returns the value of
  the tensors after the last iteration.
  - non_tensor_outputs: A dictionary containing anything that was set by
    `fn` by calling `context.set_non_tensor_output`.
</td>
</tr>

</table>



<h3 id="non_slot_devices"><code>non_slot_devices</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>non_slot_devices(
    var_list
)
</code></pre>

Device(s) for non-slot variables.

DEPRECATED: TF 1.x ONLY.

This method returns non-slot devices where non-slot variables are placed.
Users can create non-slot variables on these devices by using a block:

```python
with tf.distribute.StrategyExtended.colocate_vars_with(tf.distribute.StrategyExtended.non_slot_devices(...)):
  ...
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var_list`
</td>
<td>
The list of variables being optimized, needed with the
default <a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A sequence of devices for non-slot variables.
</td>
</tr>

</table>



<h3 id="read_var"><code>read_var</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>read_var(
    v
)
</code></pre>

Reads the value of a variable.

Returns the aggregate value of a replica-local variable, or the
(read-only) value of any other variable.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`v`
</td>
<td>
A variable allocated within the scope of this <a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tensor representing the value of `v`, aggregated across replicas if
necessary.
</td>
</tr>

</table>



<h3 id="reduce_to"><code>reduce_to</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reduce_to(
    reduce_op, value, destinations, options=None
)
</code></pre>

Combine (via e.g. sum or mean) values across replicas.

`reduce_to` aggregates <a href="../../../../tf/distribute/DistributedValues.md"><code>tf.distribute.DistributedValues</code></a> and distributed
variables. It supports both dense values and <a href="../../../../tf/IndexedSlices.md"><code>tf.IndexedSlices</code></a>.

This API currently can only be called in cross-replica context. Other
variants to reduce values across replicas are:
* <a href="../../../../tf/distribute/StrategyExtended.md#batch_reduce_to"><code>tf.distribute.StrategyExtended.batch_reduce_to</code></a>: the batch version of
  this API.
* <a href="../../../../tf/distribute/ReplicaContext.md#all_reduce"><code>tf.distribute.ReplicaContext.all_reduce</code></a>: the counterpart of this API
  in replica context. It supports both batched and non-batched all-reduce.
* <a href="../../../../tf/distribute/Strategy.md#reduce"><code>tf.distribute.Strategy.reduce</code></a>: a more convenient method to reduce
  to the host in cross-replica context.

`destinations` specifies where to reduce the value to, e.g. "GPU:0". You can
also pass in a `Tensor`, and the destinations will be the device of that
tensor. For all-reduce, pass the same to `value` and `destinations`.

It can be used in <a href="../../../../tf/distribute/ReplicaContext.md#merge_call"><code>tf.distribute.ReplicaContext.merge_call</code></a> to write code
that works for all <a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>.

```
>>> @tf.function
... def step_fn(var):
...
...   def merge_fn(strategy, value, var):
...     # All-reduce the value. Note that `value` here is a
...     # `tf.distribute.DistributedValues`.
...     reduced = strategy.extended.reduce_to(tf.distribute.ReduceOp.SUM,
...         value, destinations=var)
...     strategy.extended.update(var, lambda var, value: var.assign(value),
...         args=(reduced,))
...
...   value = tf.identity(1.)
...   tf.distribute.get_replica_context().merge_call(merge_fn,
...     args=(value, var))
>>>
>>> def run(strategy):
...   with strategy.scope():
...     v = tf.Variable(0.)
...     strategy.run(step_fn, args=(v,))
...     return v
>>>
>>> run(tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"]))
MirroredVariable:{
  0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>,
  1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=2.0>
}
>>> run(tf.distribute.experimental.CentralStorageStrategy(
...     compute_devices=["GPU:0", "GPU:1"], parameter_device="CPU:0"))
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
>>> run(tf.distribute.OneDeviceStrategy("GPU:0"))
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`reduce_op`
</td>
<td>
a <a href="../../../../tf/distribute/ReduceOp.md"><code>tf.distribute.ReduceOp</code></a> value specifying how values should
be combined. Allows using string representation of the enum such as
"SUM", "MEAN".
</td>
</tr><tr>
<td>
`value`
</td>
<td>
a <a href="../../../../tf/distribute/DistributedValues.md"><code>tf.distribute.DistributedValues</code></a>, or a <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a> like object.
</td>
</tr><tr>
<td>
`destinations`
</td>
<td>
a <a href="../../../../tf/distribute/DistributedValues.md"><code>tf.distribute.DistributedValues</code></a>, a <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a>, a
<a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a> alike object, or a device string. It specifies the devices
to reduce to. To perform an all-reduce, pass the same to `value` and
`destinations`. Note that if it's a <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a>, the value is reduced
to the devices of that variable, and this method doesn't update the
variable.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
a <a href="../../../../tf/distribute/experimental/CommunicationOptions.md"><code>tf.distribute.experimental.CommunicationOptions</code></a>. Options to
perform collective operations. This overrides the default options if the
<a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> takes one in the constructor. See
<a href="../../../../tf/distribute/experimental/CommunicationOptions.md"><code>tf.distribute.experimental.CommunicationOptions</code></a> for details of the
options.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tensor or value reduced to `destinations`.
</td>
</tr>

</table>



<h3 id="update"><code>update</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update(
    var, fn, args=(), kwargs=None, group=True
)
</code></pre>

Run `fn` to update `var` using inputs mirrored to the same devices.

<a href="../../../../tf/distribute/StrategyExtended.md#update"><code>tf.distribute.StrategyExtended.update</code></a> takes a distributed variable `var`
to be updated, an update function `fn`, and `args` and `kwargs` for `fn`. It
applies `fn` to each component variable of `var` and passes corresponding
values from `args` and `kwargs`. Neither `args` nor `kwargs` may contain
per-replica values. If they contain mirrored values, they will be unwrapped
before calling `fn`. For example, `fn` can be `assign_add` and `args` can be
a mirrored DistributedValues where each component contains the value to be
added to this mirrored variable `var`. Calling `update` will call
`assign_add` on each component variable of `var` with the corresponding
tensor value on that device.

#### Example usage:



```python
strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1']) # With 2
devices
with strategy.scope():
  v = tf.Variable(5.0, aggregation=tf.VariableAggregation.SUM)
def update_fn(v):
  return v.assign(1.0)
result = strategy.extended.update(v, update_fn)
# result is
# Mirrored:{
#  0: tf.Tensor(1.0, shape=(), dtype=float32),
#  1: tf.Tensor(1.0, shape=(), dtype=float32)
# }
```

If `var` is mirrored across multiple devices, then this method implements
logic as following:

```python
results = {}
for device, v in var:
  with tf.device(device):
    # args and kwargs will be unwrapped if they are mirrored.
    results[device] = fn(v, *args, **kwargs)
return merged(results)
```

Otherwise, this method returns `fn(var, *args, **kwargs)` colocated with
`var`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var`
</td>
<td>
Variable, possibly mirrored to multiple devices, to operate on.
</td>
</tr><tr>
<td>
`fn`
</td>
<td>
Function to call. Should take the variable as the first argument.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
Tuple or list. Additional positional arguments to pass to `fn()`.
</td>
</tr><tr>
<td>
`kwargs`
</td>
<td>
Dict with keyword arguments to pass to `fn()`.
</td>
</tr><tr>
<td>
`group`
</td>
<td>
Boolean. Defaults to True. If False, the return value will be
unwrapped.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
By default, the merged return value of `fn` across all replicas.  The
merged result has dependencies to make sure that if it is evaluated at
all, the side effects (updates) will happen on every replica. If instead
"group=False" is specified, this function will return a nest of lists
where each list has an element per replica, and the caller is responsible
for ensuring all elements are executed.
</td>
</tr>

</table>



<h3 id="update_non_slot"><code>update_non_slot</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_non_slot(
    colocate_with, fn, args=(), kwargs=None, group=True
)
</code></pre>

Runs `fn(*args, **kwargs)` on `colocate_with` devices.

Used to update non-slot variables.

DEPRECATED: TF 1.x ONLY.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`colocate_with`
</td>
<td>
Devices returned by `non_slot_devices()`.
</td>
</tr><tr>
<td>
`fn`
</td>
<td>
Function to execute.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
Tuple or list. Positional arguments to pass to `fn()`.
</td>
</tr><tr>
<td>
`kwargs`
</td>
<td>
Dict with keyword arguments to pass to `fn()`.
</td>
</tr><tr>
<td>
`group`
</td>
<td>
Boolean. Defaults to True. If False, the return value will be
unwrapped.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Return value of `fn`, possibly merged across devices.
</td>
</tr>

</table>



<h3 id="value_container"><code>value_container</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>value_container(
    value
)
</code></pre>

Returns the container that this per-replica `value` belongs to.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
A value returned by `run()` or a variable created in `scope()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A container that `value` belongs to.
If value does not belong to any container (including the case of
container having been destroyed), returns the value itself.
`value in experimental_local_results(value_container(value))` will
always be true.
</td>
</tr>

</table>



<h3 id="variable_created_in_scope"><code>variable_created_in_scope</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>variable_created_in_scope(
    v
)
</code></pre>

Tests whether `v` was created while this strategy scope was active.

Variables created inside the strategy scope are "owned" by it:

```
>>> strategy = tf.distribute.MirroredStrategy()
>>> with strategy.scope():
...   v = tf.Variable(1.)
>>> strategy.extended.variable_created_in_scope(v)
True
```

Variables created outside the strategy are not owned by it:

```
>>> strategy = tf.distribute.MirroredStrategy()
>>> v = tf.Variable(1.)
>>> strategy.extended.variable_created_in_scope(v)
False
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`v`
</td>
<td>
A <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a> instance.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True if `v` was created inside the scope, False if not.
</td>
</tr>

</table>





