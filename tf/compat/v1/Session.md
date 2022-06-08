description: A class for running TensorFlow operations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.Session" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="as_default"/>
<meta itemprop="property" content="close"/>
<meta itemprop="property" content="list_devices"/>
<meta itemprop="property" content="make_callable"/>
<meta itemprop="property" content="partial_run"/>
<meta itemprop="property" content="partial_run_setup"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="run"/>
</div>

# tf.compat.v1.Session

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>



A class for running TensorFlow operations.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.Session(
    target=&#x27;&#x27;, graph=None, config=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

`Session` does not work with either eager execution or <a href="../../../tf/function.md"><code>tf.function</code></a>, and you
should not invoke it directly. To migrate code that uses sessions to TF2,
rewrite the code without it. See the
[migration
guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
on replacing `Session.run` calls.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

A `Session` object encapsulates the environment in which `Operation`
objects are executed, and `Tensor` objects are evaluated. For
example:

```python
tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.compat.v1.Session()

# Evaluate the tensor `c`.
print(sess.run(c)) # prints 30.0
```

A session may own resources, such as
<a href="../../../tf/Variable.md"><code>tf.Variable</code></a>, <a href="../../../tf/queue/QueueBase.md"><code>tf.queue.QueueBase</code></a>,
and <a href="../../../tf/compat/v1/ReaderBase.md"><code>tf.compat.v1.ReaderBase</code></a>. It is important to release
these resources when they are no longer required. To do this, either
invoke the `tf.Session.close` method on the session, or use
the session as a context manager. The following two examples are
equivalent:

```python
# Using the `close()` method.
sess = tf.compat.v1.Session()
sess.run(...)
sess.close()

# Using the context manager.
with tf.compat.v1.Session() as sess:
  sess.run(...)
```

The
[`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
protocol buffer exposes various configuration options for a
session. For example, to create a session that uses soft constraints
for device placement, and log the resulting placement decisions,
create a session as follows:

```python
# Launch the graph in a session that allows soft device placement and
# logs the placement decisions.
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True))
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`target`
</td>
<td>
(Optional.) The execution engine to connect to. Defaults to using
an in-process engine. See
[Distributed TensorFlow](https://tensorflow.org/deploy/distributed) for
  more examples.
</td>
</tr><tr>
<td>
`graph`
</td>
<td>
(Optional.) The `Graph` to be launched (described above).
</td>
</tr><tr>
<td>
`config`
</td>
<td>
(Optional.) A
[`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
  protocol buffer with configuration options for the session.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`graph`
</td>
<td>
The graph that was launched in this session.
</td>
</tr><tr>
<td>
`graph_def`
</td>
<td>
A serializable version of the underlying TensorFlow graph.
</td>
</tr><tr>
<td>
`sess_str`
</td>
<td>
The TensorFlow process to which this session will connect.
</td>
</tr>
</table>



## Methods

<h3 id="as_default"><code>as_default</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_default()
</code></pre>

Returns a context manager that makes this object the default session.

Use with the `with` keyword to specify that calls to
<a href="../../../tf/Operation.md#run"><code>tf.Operation.run</code></a> or <a href="../../../tf/Tensor.md#eval"><code>tf.Tensor.eval</code></a> should be executed in
this session.

```python
c = tf.constant(..)
sess = tf.compat.v1.Session()

with sess.as_default():
  assert tf.compat.v1.get_default_session() is sess
  print(c.eval())
```

To get the current default session, use <a href="../../../tf/compat/v1/get_default_session.md"><code>tf.compat.v1.get_default_session</code></a>.

*N.B.* The `as_default` context manager *does not* close the
session when you exit the context, and you must close the session
explicitly.

```python
c = tf.constant(...)
sess = tf.compat.v1.Session()
with sess.as_default():
  print(c.eval())
# ...
with sess.as_default():
  print(c.eval())

sess.close()
```

Alternatively, you can use `with tf.compat.v1.Session():` to create a
session that is automatically closed on exiting the context,
including when an uncaught exception is raised.

*N.B.* The default session is a property of the current thread. If you
create a new thread, and wish to use the default session in that
thread, you must explicitly add a `with sess.as_default():` in that
thread's function.

*N.B.* Entering a `with sess.as_default():` block does not affect
the current default graph. If you are using multiple graphs, and
`sess.graph` is different from the value of
<a href="../../../tf/compat/v1/get_default_graph.md"><code>tf.compat.v1.get_default_graph</code></a>, you must explicitly enter a
`with sess.graph.as_default():` block to make `sess.graph` the default
graph.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A context manager using this session as the default session.
</td>
</tr>

</table>



<h3 id="close"><code>close</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>close()
</code></pre>

Closes this session.

Calling this method frees all resources associated with the session.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`tf.errors.OpError`
</td>
<td>
Or one of its subclasses if an error occurs while
closing the TensorFlow session.
</td>
</tr>
</table>



<h3 id="list_devices"><code>list_devices</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_devices()
</code></pre>

Lists available devices in this session.

```python
devices = sess.list_devices()
for d in devices:
  print(d.name)
```

#### Where:

Each element in the list has the following properties

* <b>`name`</b>: A string with the full name of the device. ex:
    `/job:worker/replica:0/task:3/device:CPU:0`
* <b>`device_type`</b>: The type of the device (e.g. `CPU`, `GPU`, `TPU`.)
* <b>`memory_limit`</b>: The maximum amount of memory available on the device.
    Note: depending on the device, it is possible the usable memory could
    be substantially less.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`tf.errors.OpError`
</td>
<td>
If it encounters an error (e.g. session is in an
invalid state, or network errors occur).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of devices in the session.
</td>
</tr>

</table>



<h3 id="make_callable"><code>make_callable</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>make_callable(
    fetches, feed_list=None, accept_options=False
)
</code></pre>

Returns a Python callable that runs a particular step.

The returned callable will take `len(feed_list)` arguments whose types
must be compatible feed values for the respective elements of `feed_list`.
For example, if element `i` of `feed_list` is a <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>, the `i`th
argument to the returned callable must be a numpy ndarray (or something
convertible to an ndarray) with matching element type and shape. See
`tf.Session.run` for details of the allowable feed key and value types.

The returned callable will have the same return type as
`tf.Session.run(fetches, ...)`. For example, if `fetches` is a <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>,
the callable will return a numpy ndarray; if `fetches` is a <a href="../../../tf/Operation.md"><code>tf.Operation</code></a>,
it will return `None`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fetches`
</td>
<td>
A value or list of values to fetch. See `tf.Session.run` for
details of the allowable fetch types.
</td>
</tr><tr>
<td>
`feed_list`
</td>
<td>
(Optional.) A list of `feed_dict` keys. See `tf.Session.run`
for details of the allowable feed key types.
</td>
</tr><tr>
<td>
`accept_options`
</td>
<td>
(Optional.) If `True`, the returned `Callable` will be
able to accept <a href="../../../tf/compat/v1/RunOptions.md"><code>tf.compat.v1.RunOptions</code></a> and <a href="../../../tf/compat/v1/RunMetadata.md"><code>tf.compat.v1.RunMetadata</code></a>
as optional keyword arguments `options` and `run_metadata`,
respectively, with the same syntax and semantics as `tf.Session.run`,
which is useful for certain use cases (profiling and debugging) but will
result in measurable slowdown of the `Callable`'s
performance. Default: `False`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A function that when called will execute the step defined by
`feed_list` and `fetches` in this session.
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
If `fetches` or `feed_list` cannot be interpreted
as arguments to `tf.Session.run`.
</td>
</tr>
</table>



<h3 id="partial_run"><code>partial_run</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>partial_run(
    handle, fetches, feed_dict=None
)
</code></pre>

Continues the execution with more feeds and fetches.

This is EXPERIMENTAL and subject to change.

To use partial execution, a user first calls `partial_run_setup()` and
then a sequence of `partial_run()`. `partial_run_setup` specifies the
list of feeds and fetches that will be used in the subsequent
`partial_run` calls.

The optional `feed_dict` argument allows the caller to override
the value of tensors in the graph. See run() for more information.

Below is a simple example:

```python
a = array_ops.placeholder(dtypes.float32, shape=[])
b = array_ops.placeholder(dtypes.float32, shape=[])
c = array_ops.placeholder(dtypes.float32, shape=[])
r1 = math_ops.add(a, b)
r2 = math_ops.multiply(r1, c)

h = sess.partial_run_setup([r1, r2], [a, b, c])
res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
res = sess.partial_run(h, r2, feed_dict={c: res})
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`handle`
</td>
<td>
A handle for a sequence of partial runs.
</td>
</tr><tr>
<td>
`fetches`
</td>
<td>
A single graph element, a list of graph elements, or a dictionary
whose values are graph elements or lists of graph elements (see
documentation for `run`).
</td>
</tr><tr>
<td>
`feed_dict`
</td>
<td>
A dictionary that maps graph elements to values (described
above).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Either a single value if `fetches` is a single graph element, or
a list of values if `fetches` is a list, or a dictionary with the
same keys as `fetches` if that is a dictionary
(see documentation for `run`).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`tf.errors.OpError`
</td>
<td>
Or one of its subclasses on error.
</td>
</tr>
</table>



<h3 id="partial_run_setup"><code>partial_run_setup</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>partial_run_setup(
    fetches, feeds=None
)
</code></pre>

Sets up a graph with feeds and fetches for partial run.

This is EXPERIMENTAL and subject to change.

Note that contrary to `run`, `feeds` only specifies the graph elements.
The tensors will be supplied by the subsequent `partial_run` calls.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fetches`
</td>
<td>
A single graph element, or a list of graph elements.
</td>
</tr><tr>
<td>
`feeds`
</td>
<td>
A single graph element, or a list of graph elements.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A handle for partial run.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If this `Session` is in an invalid state (e.g. has been
closed).
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If `fetches` or `feed_dict` keys are of an inappropriate type.
</td>
</tr><tr>
<td>
`tf.errors.OpError`
</td>
<td>
Or one of its subclasses if a TensorFlow error happens.
</td>
</tr>
</table>



<h3 id="reset"><code>reset</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>reset(
    target, containers=None, config=None
)
</code></pre>

Resets resource containers on `target`, and close all connected sessions.

A resource container is distributed across all workers in the
same cluster as `target`.  When a resource container on `target`
is reset, resources associated with that container will be cleared.
In particular, all Variables in the container will become undefined:
they lose their values and shapes.

#### NOTE:


(i) reset() is currently only implemented for distributed sessions.
(ii) Any sessions on the master named by `target` will be closed.

If no resource containers are provided, all containers are reset.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`target`
</td>
<td>
The execution engine to connect to.
</td>
</tr><tr>
<td>
`containers`
</td>
<td>
A list of resource container name strings, or `None` if all of
all the containers are to be reset.
</td>
</tr><tr>
<td>
`config`
</td>
<td>
(Optional.) Protocol buffer with configuration options.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`tf.errors.OpError`
</td>
<td>
Or one of its subclasses if an error occurs while
resetting containers.
</td>
</tr>
</table>



<h3 id="run"><code>run</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run(
    fetches, feed_dict=None, options=None, run_metadata=None
)
</code></pre>

Runs operations and evaluates tensors in `fetches`.

This method runs one "step" of TensorFlow computation, by
running the necessary graph fragment to execute every `Operation`
and evaluate every `Tensor` in `fetches`, substituting the values in
`feed_dict` for the corresponding input values.

The `fetches` argument may be a single graph element, or an arbitrarily
nested list, tuple, namedtuple, dict, or OrderedDict containing graph
elements at its leaves.  A graph element can be one of the following types:

* A <a href="../../../tf/Operation.md"><code>tf.Operation</code></a>.
  The corresponding fetched value will be `None`.
* A <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>.
  The corresponding fetched value will be a numpy ndarray containing the
  value of that tensor.
* A <a href="../../../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a>.
  The corresponding fetched value will be a
  <a href="../../../tf/compat/v1/SparseTensorValue.md"><code>tf.compat.v1.SparseTensorValue</code></a>
  containing the value of that sparse tensor.
* A `get_tensor_handle` op.  The corresponding fetched value will be a
  numpy ndarray containing the handle of that tensor.
* A `string` which is the name of a tensor or operation in the graph.

The value returned by `run()` has the same shape as the `fetches` argument,
where the leaves are replaced by the corresponding values returned by
TensorFlow.

#### Example:



```python
   a = tf.constant([10, 20])
   b = tf.constant([1.0, 2.0])
   # 'fetches' can be a singleton
   v = session.run(a)
   # v is the numpy array [10, 20]
   # 'fetches' can be a list.
   v = session.run([a, b])
   # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
   # 1-D array [1.0, 2.0]
   # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
   MyData = collections.namedtuple('MyData', ['a', 'b'])
   v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
   # v is a dict with
   # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
   # 'b' (the numpy array [1.0, 2.0])
   # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
   # [10, 20].
```

The optional `feed_dict` argument allows the caller to override
the value of tensors in the graph. Each key in `feed_dict` can be
one of the following types:

* If the key is a <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>, the
  value may be a Python scalar, string, list, or numpy ndarray
  that can be converted to the same `dtype` as that
  tensor. Additionally, if the key is a
  <a href="../../../tf/compat/v1/placeholder.md"><code>tf.compat.v1.placeholder</code></a>, the shape of
  the value will be checked for compatibility with the placeholder.
* If the key is a
  <a href="../../../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a>,
  the value should be a
  <a href="../../../tf/compat/v1/SparseTensorValue.md"><code>tf.compat.v1.SparseTensorValue</code></a>.
* If the key is a nested tuple of `Tensor`s or `SparseTensor`s, the value
  should be a nested tuple with the same structure that maps to their
  corresponding values as above.

Each value in `feed_dict` must be convertible to a numpy array of the dtype
of the corresponding key.

The optional `options` argument expects a [`RunOptions`] proto. The options
allow controlling the behavior of this particular step (e.g. turning tracing
on).

The optional `run_metadata` argument expects a [`RunMetadata`] proto. When
appropriate, the non-Tensor output of this step will be collected there. For
example, when users turn on tracing in `options`, the profiled info will be
collected into this argument and passed back.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fetches`
</td>
<td>
A single graph element, a list of graph elements, or a dictionary
whose values are graph elements or lists of graph elements (described
above).
</td>
</tr><tr>
<td>
`feed_dict`
</td>
<td>
A dictionary that maps graph elements to values (described
above).
</td>
</tr><tr>
<td>
`options`
</td>
<td>
A [`RunOptions`] protocol buffer
</td>
</tr><tr>
<td>
`run_metadata`
</td>
<td>
A [`RunMetadata`] protocol buffer
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Either a single value if `fetches` is a single graph element, or
a list of values if `fetches` is a list, or a dictionary with the
same keys as `fetches` if that is a dictionary (described above).
Order in which `fetches` operations are evaluated inside the call
is undefined.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If this `Session` is in an invalid state (e.g. has been
closed).
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If `fetches` or `feed_dict` keys are of an inappropriate type.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If `fetches` or `feed_dict` keys are invalid or refer to a
`Tensor` that doesn't exist.
</td>
</tr>
</table>



<h3 id="__enter__"><code>__enter__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__enter__()
</code></pre>




<h3 id="__exit__"><code>__exit__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/client/session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__exit__(
    exec_type, exec_value, exec_tb
)
</code></pre>






