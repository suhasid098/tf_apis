description: Wraps a python function and uses it as a TensorFlow op.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.numpy_function" />
<meta itemprop="path" content="Stable" />
</div>

# tf.numpy_function

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/script_ops.py">View source</a>



Wraps a python function and uses it as a TensorFlow op.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.numpy_function`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.numpy_function(
    func, inp, Tout, stateful=True, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given a python function `func` wrap this function as an operation in a
TensorFlow function. `func` must take numpy arrays as its arguments and
return numpy arrays as its outputs.

The following example creates a TensorFlow graph with `np.sinh()` as an
operation in the graph:

```
>>> def my_numpy_func(x):
...   # x will be a numpy array with the contents of the input to the
...   # tf.function
...   return np.sinh(x)
>>> @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
... def tf_function(input):
...   y = tf.numpy_function(my_numpy_func, [input], tf.float32)
...   return y * y
>>> tf_function(tf.constant(1.))
<tf.Tensor: shape=(), dtype=float32, numpy=1.3810978>
```

Comparison to <a href="../tf/py_function.md"><code>tf.py_function</code></a>:
<a href="../tf/py_function.md"><code>tf.py_function</code></a> and <a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a> are very similar, except that
<a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a> takes numpy arrays, and not <a href="../tf/Tensor.md"><code>tf.Tensor</code></a>s. If you want the
function to contain `tf.Tensors`, and have any TensorFlow operations executed
in the function be differentiable, please use <a href="../tf/py_function.md"><code>tf.py_function</code></a>.

Note: We recommend to avoid using <a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a> outside of
prototyping and experimentation due to the following known limitations:

* Calling <a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a> will acquire the Python Global Interpreter Lock
  (GIL) that allows only one thread to run at any point in time. This will
  preclude efficient parallelization and distribution of the execution of the
  program. Therefore, you are discouraged to use <a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a> outside
  of prototyping and experimentation.

* The body of the function (i.e. `func`) will not be serialized in a
  `tf.SavedModel`. Therefore, you should not use this function if you need to
  serialize your model and restore it in a different environment.

* The operation must run in the same address space as the Python program
  that calls <a href="../tf/numpy_function.md"><code>tf.numpy_function()</code></a>. If you are using distributed
  TensorFlow, you must run a <a href="../tf/distribute/Server.md"><code>tf.distribute.Server</code></a> in the same process as the
  program that calls <a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a>  you must pin the created
  operation to a device in that server (e.g. using `with tf.device():`).

* Currently <a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a> is not compatible with XLA. Calling
  <a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a> inside <a href="../tf/function.md"><code>tf.function(jit_comiple=True)</code></a> will raise an
  error.

* Since the function takes numpy arrays, you cannot take gradients
  through a numpy_function. If you require something that is differentiable,
  please consider using tf.py_function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`func`
</td>
<td>
A Python function, which accepts `numpy.ndarray` objects as arguments
and returns a list of `numpy.ndarray` objects (or a single
`numpy.ndarray`). This function must accept as many arguments as there are
tensors in `inp`, and these argument types will match the corresponding
<a href="../tf/Tensor.md"><code>tf.Tensor</code></a> objects in `inp`. The returns `numpy.ndarray`s must match the
number and types defined `Tout`.
Important Note: Input and output `numpy.ndarray`s of `func` are not
  guaranteed to be copies. In some cases their underlying memory will be
  shared with the corresponding TensorFlow tensors. In-place modification
  or storing `func` input or return values in python datastructures
  without explicit (np.)copy can have non-deterministic consequences.
</td>
</tr><tr>
<td>
`inp`
</td>
<td>
A list of <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> objects.
</td>
</tr><tr>
<td>
`Tout`
</td>
<td>
A list or tuple of tensorflow data types or a single tensorflow data
type if there is only one, indicating what `func` returns.
</td>
</tr><tr>
<td>
`stateful`
</td>
<td>
(Boolean.) Setting this argument to False tells the runtime to
treat the function as stateless, which enables certain optimizations.
A function is stateless when given the same input it will return the
same output and have no side effects; its only purpose is to have a
return value.
The behavior for a stateful function with the `stateful` argument False
is undefined. In particular, caution should be taken when
mutating the input arguments as this is a stateful operation.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(Optional) A name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Single or list of <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> which `func` computes.
</td>
</tr>

</table>

