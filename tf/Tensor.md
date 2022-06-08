description: A <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> represents a multidimensional array of elements.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.Tensor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__abs__"/>
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__and__"/>
<meta itemprop="property" content="__array__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__div__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__floordiv__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__invert__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__matmul__"/>
<meta itemprop="property" content="__mod__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__nonzero__"/>
<meta itemprop="property" content="__or__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__rand__"/>
<meta itemprop="property" content="__rdiv__"/>
<meta itemprop="property" content="__rfloordiv__"/>
<meta itemprop="property" content="__rmatmul__"/>
<meta itemprop="property" content="__rmod__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__ror__"/>
<meta itemprop="property" content="__rpow__"/>
<meta itemprop="property" content="__rsub__"/>
<meta itemprop="property" content="__rtruediv__"/>
<meta itemprop="property" content="__rxor__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="__xor__"/>
<meta itemprop="property" content="consumers"/>
<meta itemprop="property" content="eval"/>
<meta itemprop="property" content="experimental_ref"/>
<meta itemprop="property" content="get_shape"/>
<meta itemprop="property" content="ref"/>
<meta itemprop="property" content="set_shape"/>
<meta itemprop="property" content="OVERLOADABLE_OPERATORS"/>
</div>

# tf.Tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



A <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> represents a multidimensional array of elements.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.experimental.numpy.ndarray`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.Tensor`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.Tensor(
    op, value_index, dtype
)
</code></pre>



<!-- Placeholder for "Used in" -->

All elements are of a single known data type.

When writing a TensorFlow program, the main object that is
manipulated and passed around is the <a href="../tf/Tensor.md"><code>tf.Tensor</code></a>.

A <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> has the following properties:

* a single data type (float32, int32, or string, for example)
* a shape

TensorFlow supports eager execution and graph execution.  In eager
execution, operations are evaluated immediately.  In graph
execution, a computational graph is constructed for later
evaluation.

TensorFlow defaults to eager execution.  In the example below, the
matrix multiplication results are calculated immediately.

```
>>> # Compute some values using a Tensor
>>> c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
>>> d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
>>> e = tf.matmul(c, d)
>>> print(e)
tf.Tensor(
[[1. 3.]
 [3. 7.]], shape=(2, 2), dtype=float32)
```

Note that during eager execution, you may discover your `Tensors` are actually
of type `EagerTensor`.  This is an internal detail, but it does give you
access to a useful function, `numpy`:

```
>>> type(e)
<class '...ops.EagerTensor'>
>>> print(e.numpy())
  [[1. 3.]
   [3. 7.]]
```

In TensorFlow, <a href="../tf/function.md"><code>tf.function</code></a>s are a common way to define graph execution.

A Tensor's shape (that is, the rank of the Tensor and the size of
each dimension) may not always be fully known.  In <a href="../tf/function.md"><code>tf.function</code></a>
definitions, the shape may only be partially known.

Most operations produce tensors of fully-known shapes if the shapes of their
inputs are also fully known, but in some cases it's only possible to find the
shape of a tensor at execution time.

A number of specialized tensors are available: see <a href="../tf/Variable.md"><code>tf.Variable</code></a>,
<a href="../tf/constant.md"><code>tf.constant</code></a>, `tf.placeholder`, <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a>, and
<a href="../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>.

Caution: when constructing a tensor from a numpy array or pandas dataframe
the underlying buffer may be re-used:

```python
a = np.array([1, 2, 3])
b = tf.constant(a)
a[0] = 4
print(b)  # tf.Tensor([4 2 3], shape=(3,), dtype=int64)
```

Note: this is an implementation detail that is subject to change and users
should not rely on this behaviour.

For more on Tensors, see the [guide](https://tensorflow.org/guide/tensor).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`op`
</td>
<td>
An `Operation`. `Operation` that computes this tensor.
</td>
</tr><tr>
<td>
`value_index`
</td>
<td>
An `int`. Index of the operation's endpoint that produces
this tensor.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
A `DType`. Type of elements stored in this tensor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If the op is not an `Operation`.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
The name of the device on which this tensor will be produced, or None.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The `DType` of elements in this tensor.
</td>
</tr><tr>
<td>
`graph`
</td>
<td>
The `Graph` that contains this tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The string name of this tensor.
</td>
</tr><tr>
<td>
`op`
</td>
<td>
The `Operation` that produces this tensor as an output.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
Returns a <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> that represents the shape of this tensor.

```
>>> t = tf.constant([1,2,3,4,5])
>>> t.shape
TensorShape([5])
```

<a href="../tf/Tensor.md#shape"><code>tf.Tensor.shape</code></a> is equivalent to <a href="../tf/Tensor.md#get_shape"><code>tf.Tensor.get_shape()</code></a>.

In a <a href="../tf/function.md"><code>tf.function</code></a> or when building a model using
<a href="../tf/keras/Input.md"><code>tf.keras.Input</code></a>, they return the build-time shape of the
tensor, which may be partially unknown.

A <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> is not a tensor. Use <a href="../tf/shape.md"><code>tf.shape(t)</code></a> to get a tensor
containing the shape, calculated at runtime.

See <a href="../tf/Tensor.md#get_shape"><code>tf.Tensor.get_shape()</code></a>, and <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> for details and examples.
</td>
</tr><tr>
<td>
`value_index`
</td>
<td>
The index of this tensor in the outputs of its `Operation`.
</td>
</tr>
</table>



## Methods

<h3 id="consumers"><code>consumers</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>consumers()
</code></pre>

Returns a list of `Operation`s that consume this tensor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of `Operation`s.
</td>
</tr>

</table>



<h3 id="eval"><code>eval</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>eval(
    feed_dict=None, session=None
)
</code></pre>

Evaluates this tensor in a `Session`.

Note: If you are not using <a href="../tf/compat/v1.md"><code>compat.v1</code></a> libraries, you should not need this,
(or `feed_dict` or `Session`).  In eager execution (or within <a href="../tf/function.md"><code>tf.function</code></a>)
you do not need to call `eval`.

Calling this method will execute all preceding operations that
produce the inputs needed for the operation that produces this
tensor.

*N.B.* Before invoking <a href="../tf/Tensor.md#eval"><code>Tensor.eval()</code></a>, its graph must have been
launched in a session, and either a default session must be
available, or `session` must be specified explicitly.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`feed_dict`
</td>
<td>
A dictionary that maps `Tensor` objects to feed values. See
`tf.Session.run` for a description of the valid feed values.
</td>
</tr><tr>
<td>
`session`
</td>
<td>
(Optional.) The `Session` to be used to evaluate this tensor. If
none, the default session will be used.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A numpy array corresponding to the value of this tensor.
</td>
</tr>

</table>



<h3 id="experimental_ref"><code>experimental_ref</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_ref()
</code></pre>

DEPRECATED FUNCTION

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use ref() instead.

<h3 id="get_shape"><code>get_shape</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_shape()
</code></pre>

Returns a <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> that represents the shape of this tensor.

In eager execution the shape is always fully-known.

```
>>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> print(a.shape)
(2, 3)
```

<a href="../tf/Tensor.md#get_shape"><code>tf.Tensor.get_shape()</code></a> is equivalent to <a href="../tf/Tensor.md#shape"><code>tf.Tensor.shape</code></a>.


When executing in a <a href="../tf/function.md"><code>tf.function</code></a> or building a model using
<a href="../tf/keras/Input.md"><code>tf.keras.Input</code></a>, <a href="../tf/Tensor.md#shape"><code>Tensor.shape</code></a> may return a partial shape (including
`None` for unknown dimensions). See <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> for more details.

```
>>> inputs = tf.keras.Input(shape = [10])
>>> # Unknown batch size
>>> print(inputs.shape)
(None, 10)
```

The shape is computed using shape inference functions that are
registered for each <a href="../tf/Operation.md"><code>tf.Operation</code></a>.

The returned <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> is determined at *build* time, without
executing the underlying kernel. It is not a <a href="../tf/Tensor.md"><code>tf.Tensor</code></a>. If you need a
shape *tensor*, either convert the <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> to a <a href="../tf/constant.md"><code>tf.constant</code></a>, or
use the <a href="../tf/shape.md"><code>tf.shape(tensor)</code></a> function, which returns the tensor's shape at
*execution* time.

This is useful for debugging and providing early errors. For
example, when tracing a <a href="../tf/function.md"><code>tf.function</code></a>, no ops are being executed, shapes
may be unknown (See the [Concrete Functions
Guide](https://www.tensorflow.org/guide/concrete_function) for details).

```
>>> @tf.function
... def my_matmul(a, b):
...   result = a@b
...   # the `print` executes during tracing.
...   print("Result shape: ", result.shape)
...   return result
```

The shape inference functions propagate shapes to the extent possible:

```
>>> f = my_matmul.get_concrete_function(
...   tf.TensorSpec([None,3]),
...   tf.TensorSpec([3,5]))
Result shape: (None, 5)
```

Tracing may fail if a shape missmatch can be detected:

```
>>> cf = my_matmul.get_concrete_function(
...   tf.TensorSpec([None,3]),
...   tf.TensorSpec([4,5]))
Traceback (most recent call last):
...
ValueError: Dimensions must be equal, but are 3 and 4 for 'matmul' (op:
'MatMul') with input shapes: [?,3], [4,5].
```

In some cases, the inferred shape may have unknown dimensions. If
the caller has additional information about the values of these
dimensions, <a href="../tf/ensure_shape.md"><code>tf.ensure_shape</code></a> or <a href="../tf/Tensor.md#set_shape"><code>Tensor.set_shape()</code></a> can be used to augment
the inferred shape.

```
>>> @tf.function
... def my_fun(a):
...   a = tf.ensure_shape(a, [5, 5])
...   # the `print` executes during tracing.
...   print("Result shape: ", a.shape)
...   return a
```

```
>>> cf = my_fun.get_concrete_function(
...   tf.TensorSpec([None, None]))
Result shape: (5, 5)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> representing the shape of this tensor.
</td>
</tr>

</table>



<h3 id="ref"><code>ref</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>ref()
</code></pre>

Returns a hashable reference object to this Tensor.

The primary use case for this API is to put tensors in a set/dictionary.
We can't put tensors in a set/dictionary as `tensor.__hash__()` is no longer
available starting Tensorflow 2.0.

The following will raise an exception starting 2.0

```
>>> x = tf.constant(5)
>>> y = tf.constant(10)
>>> z = tf.constant(10)
>>> tensor_set = {x, y, z}
Traceback (most recent call last):
  ...
TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
>>> tensor_dict = {x: 'five', y: 'ten'}
Traceback (most recent call last):
  ...
TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
```

Instead, we can use `tensor.ref()`.

```
>>> tensor_set = {x.ref(), y.ref(), z.ref()}
>>> x.ref() in tensor_set
True
>>> tensor_dict = {x.ref(): 'five', y.ref(): 'ten', z.ref(): 'ten'}
>>> tensor_dict[y.ref()]
'ten'
```

Also, the reference object provides `.deref()` function that returns the
original Tensor.

```
>>> x = tf.constant(5)
>>> x.ref().deref()
<tf.Tensor: shape=(), dtype=int32, numpy=5>
```

<h3 id="set_shape"><code>set_shape</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_shape(
    shape
)
</code></pre>

Updates the shape of this tensor.

Note: It is recommended to use <a href="../tf/ensure_shape.md"><code>tf.ensure_shape</code></a> instead of
<a href="../tf/Tensor.md#set_shape"><code>Tensor.set_shape</code></a>, because <a href="../tf/ensure_shape.md"><code>tf.ensure_shape</code></a> provides better checking for
programming errors and can create guarantees for compiler
optimization.

With eager execution this operates as a shape assertion.
Here the shapes match:

```
>>> t = tf.constant([[1,2,3]])
>>> t.set_shape([1, 3])
```

Passing a `None` in the new shape allows any value for that axis:

```
>>> t.set_shape([1,None])
```

An error is raised if an incompatible shape is passed.

```
>>> t.set_shape([1,5])
Traceback (most recent call last):
...
ValueError: Tensor's shape (1, 3) is not compatible with supplied
shape [1, 5]
```

When executing in a <a href="../tf/function.md"><code>tf.function</code></a>, or building a model using
<a href="../tf/keras/Input.md"><code>tf.keras.Input</code></a>, <a href="../tf/Tensor.md#set_shape"><code>Tensor.set_shape</code></a> will *merge* the given `shape` with
the current shape of this tensor, and set the tensor's shape to the
merged value (see <a href="../tf/TensorShape.md#merge_with"><code>tf.TensorShape.merge_with</code></a> for details):

```
>>> t = tf.keras.Input(shape=[None, None, 3])
>>> print(t.shape)
(None, None, None, 3)
```

Dimensions set to `None` are not updated:

```
>>> t.set_shape([None, 224, 224, None])
>>> print(t.shape)
(None, 224, 224, 3)
```

The main use case for this is to provide additional shape information
that cannot be inferred from the graph alone.

For example if you know all the images in a dataset have shape [28,28,3] you
can set it with `tf.set_shape`:

```
>>> @tf.function
... def load_image(filename):
...   raw = tf.io.read_file(filename)
...   image = tf.image.decode_png(raw, channels=3)
...   # the `print` executes during tracing.
...   print("Initial shape: ", image.shape)
...   image.set_shape([28, 28, 3])
...   print("Final shape: ", image.shape)
...   return image
```

Trace the function, see the [Concrete Functions
Guide](https://www.tensorflow.org/guide/concrete_function) for details.

```
>>> cf = load_image.get_concrete_function(
...     tf.TensorSpec([], dtype=tf.string))
Initial shape:  (None, None, 3)
Final shape: (28, 28, 3)
```

Similarly the <a href="../tf/io/parse_tensor.md"><code>tf.io.parse_tensor</code></a> function could return a tensor with
any shape, even the <a href="../tf/rank.md"><code>tf.rank</code></a> is unknown. If you know that all your
serialized tensors will be 2d, set it with `set_shape`:

```
>>> @tf.function
... def my_parse(string_tensor):
...   result = tf.io.parse_tensor(string_tensor, out_type=tf.float32)
...   # the `print` executes during tracing.
...   print("Initial shape: ", result.shape)
...   result.set_shape([None, None])
...   print("Final shape: ", result.shape)
...   return result
```

Trace the function

```
>>> concrete_parse = my_parse.get_concrete_function(
...     tf.TensorSpec([], dtype=tf.string))
Initial shape:  <unknown>
Final shape:  (None, None)
```

#### Make sure it works:



```
>>> t = tf.ones([5,3], dtype=tf.float32)
>>> serialized = tf.io.serialize_tensor(t)
>>> print(serialized.dtype)
<dtype: 'string'>
>>> print(serialized.shape)
()
>>> t2 = concrete_parse(serialized)
>>> print(t2.shape)
(5, 3)
```

Caution: `set_shape` ensures that the applied shape is compatible with
the existing shape, but it does not check at runtime. Setting
incorrect shapes can result in inconsistencies between the
statically-known graph and the runtime value of tensors. For runtime
validation of the shape, use <a href="../tf/ensure_shape.md"><code>tf.ensure_shape</code></a> instead. It also modifies
the `shape` of the tensor.

```
>>> # Serialize a rank-3 tensor
>>> t = tf.ones([5,5,5], dtype=tf.float32)
>>> serialized = tf.io.serialize_tensor(t)
>>> # The function still runs, even though it `set_shape([None,None])`
>>> t2 = concrete_parse(serialized)
>>> print(t2.shape)
(5, 5, 5)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`shape`
</td>
<td>
A `TensorShape` representing the shape of this tensor, a
`TensorShapeProto`, a list, a tuple, or None.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `shape` is not compatible with the current shape of
this tensor.
</td>
</tr>
</table>



<h3 id="__abs__"><code>__abs__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__abs__(
    name=None
)
</code></pre>

Computes the absolute value of a tensor.

Given a tensor of integer or floating-point values, this operation returns a
tensor of the same type, where each element contains the absolute value of the
corresponding element in the input.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float32` or `float64` that is the absolute value of each element in `x`. For
a complex number \\(a + bj\\), its absolute value is computed as
\\(\sqrt{a^2 + b^2}\\).

#### For example:



```
>>> # real number
>>> x = tf.constant([-2.25, 3.25])
>>> tf.abs(x)
<tf.Tensor: shape=(2,), dtype=float32,
numpy=array([2.25, 3.25], dtype=float32)>
```

```
>>> # complex number
>>> x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
>>> tf.abs(x)
<tf.Tensor: shape=(2, 1), dtype=float64, numpy=
array([[5.25594901],
       [6.60492241]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor` or `SparseTensor` of type `float16`, `float32`, `float64`,
`int32`, `int64`, `complex64` or `complex128`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` or `SparseTensor` of the same size, type and sparsity as `x`,
  with absolute values. Note, for `complex64` or `complex128` input, the
  returned `Tensor` will be of type `float32` or `float64`, respectively.

If `x` is a `SparseTensor`, returns
`SparseTensor(x.indices, tf.math.abs(x.values, ...), x.dense_shape)`
</td>
</tr>

</table>



<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    y
)
</code></pre>

The operation invoked by the <a href="../tf/Tensor.md#__add__"><code>Tensor.__add__</code></a> operator.


#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../tf/Tensor.md#__add__"><code>Tensor.__add__</code></a> to allow it to handle
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
The left-hand side of the `+` operator.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
The right-hand side of the `+` operator.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
an optional name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The result of the elementwise `+` operation.
</td>
</tr>

</table>



<h3 id="__and__"><code>__and__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__and__(
    y
)
</code></pre>




<h3 id="__array__"><code>__array__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__array__(
    dtype=None
)
</code></pre>




<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>

Dummy method to prevent a tensor from being used as a Python `bool`.

This overload raises a `TypeError` when the user inadvertently
treats a `Tensor` as a boolean (most commonly in an `if` or `while`
statement), in code that was not converted by AutoGraph. For example:

```python
if tf.constant(True):  # Will raise.
  # ...

if tf.constant(5) < tf.constant(7):  # Will raise.
  # ...
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
`TypeError`.
</td>
</tr>

</table>



<h3 id="__div__"><code>__div__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__div__(
    y
)
</code></pre>

Divides x / y elementwise (using Python 2 division operator semantics). (deprecated)

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.




This function divides `x` and `y`, forcing Python 2 semantics. That is, if `x`
and `y` are both integers then the result will be an integer. This is in
contrast to Python 3, where division with `/` is always a float while division
with `//` is always an integer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
`Tensor` numerator of real numeric type.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
`Tensor` denominator of real numeric type.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`x / y` returns the quotient of x and y.
</td>
</tr>

</table>



 <section><devsite-expandable >
 <h4 class="showalways">Migrate to TF2</h4>

This function is deprecated in TF2. Prefer using the Tensor division operator,
<a href="../tf/math/divide.md"><code>tf.divide</code></a>, or <a href="../tf/math/divide.md"><code>tf.math.divide</code></a>, which obey the Python 3 division operator
semantics.


 </devsite-expandable></section>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

The operation invoked by the <a href="../tf/RaggedTensor.md#__eq__"><code>Tensor.__eq__</code></a> operator.

Compares two tensors element-wise for equality if they are
broadcast-compatible; or returns False if they are not broadcast-compatible.
(Note that this behavior differs from <a href="../tf/math/equal.md"><code>tf.math.equal</code></a>, which raises an
exception if the two tensors are not broadcast-compatible.)

#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../tf/RaggedTensor.md#__eq__"><code>Tensor.__eq__</code></a> to allow it to handle
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`self`
</td>
<td>
The left-hand side of the `==` operator.
</td>
</tr><tr>
<td>
`other`
</td>
<td>
The right-hand side of the `==` operator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The result of the elementwise `==` operation, or `False` if the arguments
are not broadcast-compatible.
</td>
</tr>

</table>



<h3 id="__floordiv__"><code>__floordiv__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__floordiv__(
    y
)
</code></pre>

Divides `x / y` elementwise, rounding toward the most negative integer.

Mathematically, this is equivalent to floor(x / y). For example:
  floor(8.4 / 4.0) = floor(2.1) = 2.0
  floor(-8.4 / 4.0) = floor(-2.1) = -3.0
This is equivalent to the '//' operator in Python 3.0 and above.

Note: `x` and `y` must have the same type, and the result will have the same
type as well.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
`Tensor` numerator of real numeric type.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
`Tensor` denominator of real numeric type.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`x / y` rounded toward -infinity.
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
If the inputs are complex.
</td>
</tr>
</table>



<h3 id="__ge__"><code>__ge__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    y, name=None
)
</code></pre>

Returns the truth value of (x >= y) element-wise.

*NOTE*: <a href="../tf/math/greater_equal.md"><code>math.greater_equal</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

#### Example:



```python
x = tf.constant([5, 4, 6, 7])
y = tf.constant([5, 2, 5, 10])
tf.math.greater_equal(x, y) ==> [True, True, True, False]

x = tf.constant([5, 4, 6, 7])
y = tf.constant([5])
tf.math.greater_equal(x, y) ==> [True, False, True, True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `bool`.
</td>
</tr>

</table>



<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    slice_spec, var=None
)
</code></pre>

Overload for Tensor.__getitem__.

This operation extracts the specified region from the tensor.
The notation is similar to NumPy with the restriction that
currently only support basic indexing. That means that
using a non-scalar tensor as input is not currently allowed.

#### Some useful examples:



```python
# Strip leading and trailing 2 elements
foo = tf.constant([1,2,3,4,5,6])
print(foo[2:-2].eval())  # => [3,4]

# Skip every other row and reverse the order of the columns
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[::2,::-1].eval())  # => [[3,2,1], [9,8,7]]

# Use scalar tensors as indices on both dimensions
print(foo[tf.constant(0), tf.constant(2)].eval())  # => 3

# Insert another dimension
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[tf.newaxis, :, :].eval()) # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[:, tf.newaxis, :].eval()) # => [[[1,2,3]], [[4,5,6]], [[7,8,9]]]
print(foo[:, :, tf.newaxis].eval()) # => [[[1],[2],[3]], [[4],[5],[6]],
[[7],[8],[9]]]

# Ellipses (3 equivalent operations)
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[tf.newaxis, :, :].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[tf.newaxis, ...].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[tf.newaxis].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]

# Masks
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[foo > 2].eval())  # => [3, 4, 5, 6, 7, 8, 9]
```

#### Notes:

- <a href="../tf.md#newaxis"><code>tf.newaxis</code></a> is `None` as in NumPy.
- An implicit ellipsis is placed at the end of the `slice_spec`
- NumPy advanced indexing is currently not supported.



#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../tf/Tensor.md#__getitem__"><code>Tensor.__getitem__</code></a> to allow it to handle
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensor`
</td>
<td>
An ops.Tensor object.
</td>
</tr><tr>
<td>
`slice_spec`
</td>
<td>
The arguments to Tensor.__getitem__.
</td>
</tr><tr>
<td>
`var`
</td>
<td>
In the case of variable slice assignment, the Variable object to slice
(i.e. tensor is the read-only view of this variable).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The appropriate slice of "tensor", based on "slice_spec".
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If a slice range is negative size.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If the slice indices aren't int, slice, ellipsis,
tf.newaxis or scalar int32/int64 tensors.
</td>
</tr>
</table>



<h3 id="__gt__"><code>__gt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    y, name=None
)
</code></pre>

Returns the truth value of (x > y) element-wise.

*NOTE*: <a href="../tf/math/greater.md"><code>math.greater</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

#### Example:



```python
x = tf.constant([5, 4, 6])
y = tf.constant([5, 2, 5])
tf.math.greater(x, y) ==> [False, True, True]

x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.greater(x, y) ==> [False, False, True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `bool`.
</td>
</tr>

</table>



<h3 id="__invert__"><code>__invert__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__invert__(
    name=None
)
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>




<h3 id="__le__"><code>__le__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    y, name=None
)
</code></pre>

Returns the truth value of (x <= y) element-wise.

*NOTE*: <a href="../tf/math/less_equal.md"><code>math.less_equal</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

#### Example:



```python
x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.less_equal(x, y) ==> [True, True, False]

x = tf.constant([5, 4, 6])
y = tf.constant([5, 6, 6])
tf.math.less_equal(x, y) ==> [True, True, True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `bool`.
</td>
</tr>

</table>



<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>




<h3 id="__lt__"><code>__lt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    y, name=None
)
</code></pre>

Returns the truth value of (x < y) element-wise.

*NOTE*: <a href="../tf/math/less.md"><code>math.less</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

#### Example:



```python
x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.less(x, y) ==> [False, True, False]

x = tf.constant([5, 4, 6])
y = tf.constant([5, 6, 7])
tf.math.less(x, y) ==> [False, True, True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `bool`.
</td>
</tr>

</table>



<h3 id="__matmul__"><code>__matmul__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__matmul__(
    y
)
</code></pre>

Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

The inputs must, following any transpositions, be tensors of rank >= 2
where the inner 2 dimensions specify valid matrix multiplication dimensions,
and any further outer dimensions specify matching batch size.

Both matrices must be of the same type. The supported types are:
`bfloat16`, `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, `complex128`.

Either matrix can be transposed or adjointed (conjugated and transposed) on
the fly by setting one of the corresponding flag to `True`. These are `False`
by default.

If one or both of the matrices contain a lot of zeros, a more efficient
multiplication algorithm can be used by setting the corresponding
`a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
This optimization is only available for plain matrices (rank-2 tensors) with
datatypes `bfloat16` or `float32`.

A simple 2-D tensor matrix multiplication:

```
>>> a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
>>> a  # 2-D tensor
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)>
>>> b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
>>> b  # 2-D tensor
<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
array([[ 7,  8],
       [ 9, 10],
       [11, 12]], dtype=int32)>
>>> c = tf.matmul(a, b)
>>> c  # `a` * `b`
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[ 58,  64],
       [139, 154]], dtype=int32)>
```

A batch matrix multiplication with batch shape [2]:

```
>>> a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
>>> a  # 3-D tensor
<tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
array([[[ 1,  2,  3],
        [ 4,  5,  6]],
       [[ 7,  8,  9],
        [10, 11, 12]]], dtype=int32)>
>>> b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
>>> b  # 3-D tensor
<tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
array([[[13, 14],
        [15, 16],
        [17, 18]],
       [[19, 20],
        [21, 22],
        [23, 24]]], dtype=int32)>
>>> c = tf.matmul(a, b)
>>> c  # `a` * `b`
<tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
array([[[ 94, 100],
        [229, 244]],
       [[508, 532],
        [697, 730]]], dtype=int32)>
```

Since python >= 3.5 the @ operator is supported
(see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
it simply calls the <a href="../tf/linalg/matmul.md"><code>tf.matmul()</code></a> function, so the following lines are
equivalent:

```
>>> d = a @ b @ [[10], [11]]
>>> d = tf.matmul(tf.matmul(a, b), [[10], [11]])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`a`
</td>
<td>
<a href="../tf/Tensor.md"><code>tf.Tensor</code></a> of type `float16`, `float32`, `float64`, `int32`,
`complex64`, `complex128` and rank > 1.
</td>
</tr><tr>
<td>
`b`
</td>
<td>
<a href="../tf/Tensor.md"><code>tf.Tensor</code></a> with same type and rank as `a`.
</td>
</tr><tr>
<td>
`transpose_a`
</td>
<td>
If `True`, `a` is transposed before multiplication.
</td>
</tr><tr>
<td>
`transpose_b`
</td>
<td>
If `True`, `b` is transposed before multiplication.
</td>
</tr><tr>
<td>
`adjoint_a`
</td>
<td>
If `True`, `a` is conjugated and transposed before
multiplication.
</td>
</tr><tr>
<td>
`adjoint_b`
</td>
<td>
If `True`, `b` is conjugated and transposed before
multiplication.
</td>
</tr><tr>
<td>
`a_is_sparse`
</td>
<td>
If `True`, `a` is treated as a sparse matrix. Notice, this
**does not support <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a>**, it just makes optimizations
that assume most values in `a` are zero.
See <a href="../tf/sparse/sparse_dense_matmul.md"><code>tf.sparse.sparse_dense_matmul</code></a>
for some support for <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> multiplication.
</td>
</tr><tr>
<td>
`b_is_sparse`
</td>
<td>
If `True`, `b` is treated as a sparse matrix. Notice, this
**does not support <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a>**, it just makes optimizations
that assume most values in `a` are zero.
See <a href="../tf/sparse/sparse_dense_matmul.md"><code>tf.sparse.sparse_dense_matmul</code></a>
for some support for <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> multiplication.
</td>
</tr><tr>
<td>
`output_type`
</td>
<td>
The output datatype if needed. Defaults to None in which case
the output_type is the same as input type. Currently only works when input
tensors are type (u)int8 and output_type can be int32.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> of the same type as `a` and `b` where each inner-most matrix
is the product of the corresponding matrices in `a` and `b`, e.g. if all
transpose or adjoint attributes are `False`:

`output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
for all indices `i`, `j`.
</td>
</tr>
<tr>
<td>
`Note`
</td>
<td>
This is matrix product, not element-wise product.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `transpose_a` and `adjoint_a`, or `transpose_b` and
`adjoint_b` are both set to `True`.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If output_type is specified but the types of `a`, `b` and
`output_type` is not (u)int8, (u)int8 and int32.
</td>
</tr>
</table>



<h3 id="__mod__"><code>__mod__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mod__(
    y
)
</code></pre>

Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: <a href="../tf/math/floormod.md"><code>math.floormod</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bfloat16`, `half`, `float32`, `float64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `x`.
</td>
</tr>

</table>



<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    y
)
</code></pre>

Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse".


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

The operation invoked by the <a href="../tf/RaggedTensor.md#__ne__"><code>Tensor.__ne__</code></a> operator.

Compares two tensors element-wise for inequality if they are
broadcast-compatible; or returns True if they are not broadcast-compatible.
(Note that this behavior differs from <a href="../tf/math/not_equal.md"><code>tf.math.not_equal</code></a>, which raises an
exception if the two tensors are not broadcast-compatible.)

#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../tf/RaggedTensor.md#__ne__"><code>Tensor.__ne__</code></a> to allow it to handle
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`self`
</td>
<td>
The left-hand side of the `!=` operator.
</td>
</tr><tr>
<td>
`other`
</td>
<td>
The right-hand side of the `!=` operator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The result of the elementwise `!=` operation, or `True` if the arguments
are not broadcast-compatible.
</td>
</tr>

</table>



<h3 id="__neg__"><code>__neg__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__(
    name=None
)
</code></pre>

Computes numerical negative value element-wise.

I.e., \\(y = -x\\).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `x`.

If `x` is a `SparseTensor`, returns
`SparseTensor(x.indices, tf.math.negative(x.values, ...), x.dense_shape)`
</td>
</tr>

</table>



<h3 id="__nonzero__"><code>__nonzero__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__nonzero__()
</code></pre>

Dummy method to prevent a tensor from being used as a Python `bool`.

This is the Python 2.x counterpart to `__bool__()` above.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
`TypeError`.
</td>
</tr>

</table>



<h3 id="__or__"><code>__or__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__or__(
    y
)
</code></pre>




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    y
)
</code></pre>

Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```python
x = tf.constant([[2, 2], [3, 3]])
y = tf.constant([[8, 16], [2, 3]])
tf.pow(x, y)  # [[256, 65536], [9, 27]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, or `complex128`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, or `complex128`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`.
</td>
</tr>

</table>



<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    x
)
</code></pre>

The operation invoked by the <a href="../tf/Tensor.md#__add__"><code>Tensor.__add__</code></a> operator.


#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../tf/Tensor.md#__add__"><code>Tensor.__add__</code></a> to allow it to handle
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
The left-hand side of the `+` operator.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
The right-hand side of the `+` operator.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
an optional name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The result of the elementwise `+` operation.
</td>
</tr>

</table>



<h3 id="__rand__"><code>__rand__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rand__(
    x
)
</code></pre>




<h3 id="__rdiv__"><code>__rdiv__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rdiv__(
    x
)
</code></pre>

Divides x / y elementwise (using Python 2 division operator semantics). (deprecated)

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.




This function divides `x` and `y`, forcing Python 2 semantics. That is, if `x`
and `y` are both integers then the result will be an integer. This is in
contrast to Python 3, where division with `/` is always a float while division
with `//` is always an integer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
`Tensor` numerator of real numeric type.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
`Tensor` denominator of real numeric type.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`x / y` returns the quotient of x and y.
</td>
</tr>

</table>



 <section><devsite-expandable >
 <h4 class="showalways">Migrate to TF2</h4>

This function is deprecated in TF2. Prefer using the Tensor division operator,
<a href="../tf/math/divide.md"><code>tf.divide</code></a>, or <a href="../tf/math/divide.md"><code>tf.math.divide</code></a>, which obey the Python 3 division operator
semantics.


 </devsite-expandable></section>



<h3 id="__rfloordiv__"><code>__rfloordiv__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rfloordiv__(
    x
)
</code></pre>

Divides `x / y` elementwise, rounding toward the most negative integer.

Mathematically, this is equivalent to floor(x / y). For example:
  floor(8.4 / 4.0) = floor(2.1) = 2.0
  floor(-8.4 / 4.0) = floor(-2.1) = -3.0
This is equivalent to the '//' operator in Python 3.0 and above.

Note: `x` and `y` must have the same type, and the result will have the same
type as well.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
`Tensor` numerator of real numeric type.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
`Tensor` denominator of real numeric type.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`x / y` rounded toward -infinity.
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
If the inputs are complex.
</td>
</tr>
</table>



<h3 id="__rmatmul__"><code>__rmatmul__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmatmul__(
    x
)
</code></pre>

Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

The inputs must, following any transpositions, be tensors of rank >= 2
where the inner 2 dimensions specify valid matrix multiplication dimensions,
and any further outer dimensions specify matching batch size.

Both matrices must be of the same type. The supported types are:
`bfloat16`, `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, `complex128`.

Either matrix can be transposed or adjointed (conjugated and transposed) on
the fly by setting one of the corresponding flag to `True`. These are `False`
by default.

If one or both of the matrices contain a lot of zeros, a more efficient
multiplication algorithm can be used by setting the corresponding
`a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
This optimization is only available for plain matrices (rank-2 tensors) with
datatypes `bfloat16` or `float32`.

A simple 2-D tensor matrix multiplication:

```
>>> a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
>>> a  # 2-D tensor
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)>
>>> b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
>>> b  # 2-D tensor
<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
array([[ 7,  8],
       [ 9, 10],
       [11, 12]], dtype=int32)>
>>> c = tf.matmul(a, b)
>>> c  # `a` * `b`
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[ 58,  64],
       [139, 154]], dtype=int32)>
```

A batch matrix multiplication with batch shape [2]:

```
>>> a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
>>> a  # 3-D tensor
<tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
array([[[ 1,  2,  3],
        [ 4,  5,  6]],
       [[ 7,  8,  9],
        [10, 11, 12]]], dtype=int32)>
>>> b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
>>> b  # 3-D tensor
<tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
array([[[13, 14],
        [15, 16],
        [17, 18]],
       [[19, 20],
        [21, 22],
        [23, 24]]], dtype=int32)>
>>> c = tf.matmul(a, b)
>>> c  # `a` * `b`
<tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
array([[[ 94, 100],
        [229, 244]],
       [[508, 532],
        [697, 730]]], dtype=int32)>
```

Since python >= 3.5 the @ operator is supported
(see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
it simply calls the <a href="../tf/linalg/matmul.md"><code>tf.matmul()</code></a> function, so the following lines are
equivalent:

```
>>> d = a @ b @ [[10], [11]]
>>> d = tf.matmul(tf.matmul(a, b), [[10], [11]])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`a`
</td>
<td>
<a href="../tf/Tensor.md"><code>tf.Tensor</code></a> of type `float16`, `float32`, `float64`, `int32`,
`complex64`, `complex128` and rank > 1.
</td>
</tr><tr>
<td>
`b`
</td>
<td>
<a href="../tf/Tensor.md"><code>tf.Tensor</code></a> with same type and rank as `a`.
</td>
</tr><tr>
<td>
`transpose_a`
</td>
<td>
If `True`, `a` is transposed before multiplication.
</td>
</tr><tr>
<td>
`transpose_b`
</td>
<td>
If `True`, `b` is transposed before multiplication.
</td>
</tr><tr>
<td>
`adjoint_a`
</td>
<td>
If `True`, `a` is conjugated and transposed before
multiplication.
</td>
</tr><tr>
<td>
`adjoint_b`
</td>
<td>
If `True`, `b` is conjugated and transposed before
multiplication.
</td>
</tr><tr>
<td>
`a_is_sparse`
</td>
<td>
If `True`, `a` is treated as a sparse matrix. Notice, this
**does not support <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a>**, it just makes optimizations
that assume most values in `a` are zero.
See <a href="../tf/sparse/sparse_dense_matmul.md"><code>tf.sparse.sparse_dense_matmul</code></a>
for some support for <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> multiplication.
</td>
</tr><tr>
<td>
`b_is_sparse`
</td>
<td>
If `True`, `b` is treated as a sparse matrix. Notice, this
**does not support <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a>**, it just makes optimizations
that assume most values in `a` are zero.
See <a href="../tf/sparse/sparse_dense_matmul.md"><code>tf.sparse.sparse_dense_matmul</code></a>
for some support for <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> multiplication.
</td>
</tr><tr>
<td>
`output_type`
</td>
<td>
The output datatype if needed. Defaults to None in which case
the output_type is the same as input type. Currently only works when input
tensors are type (u)int8 and output_type can be int32.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> of the same type as `a` and `b` where each inner-most matrix
is the product of the corresponding matrices in `a` and `b`, e.g. if all
transpose or adjoint attributes are `False`:

`output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
for all indices `i`, `j`.
</td>
</tr>
<tr>
<td>
`Note`
</td>
<td>
This is matrix product, not element-wise product.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `transpose_a` and `adjoint_a`, or `transpose_b` and
`adjoint_b` are both set to `True`.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If output_type is specified but the types of `a`, `b` and
`output_type` is not (u)int8, (u)int8 and int32.
</td>
</tr>
</table>



<h3 id="__rmod__"><code>__rmod__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmod__(
    x
)
</code></pre>

Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: <a href="../tf/math/floormod.md"><code>math.floormod</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bfloat16`, `half`, `float32`, `float64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `x`.
</td>
</tr>

</table>



<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    x
)
</code></pre>

Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse".


<h3 id="__ror__"><code>__ror__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ror__(
    x
)
</code></pre>




<h3 id="__rpow__"><code>__rpow__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rpow__(
    x
)
</code></pre>

Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```python
x = tf.constant([[2, 2], [3, 3]])
y = tf.constant([[8, 16], [2, 3]])
tf.pow(x, y)  # [[256, 65536], [9, 27]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, or `complex128`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, or `complex128`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`.
</td>
</tr>

</table>



<h3 id="__rsub__"><code>__rsub__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rsub__(
    x
)
</code></pre>

Returns x - y element-wise.

*NOTE*: <a href="../tf/math/subtract.md"><code>tf.subtract</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Both input and output have a range `(-inf, inf)`.

Example usages below.

Subtract operation between an array and a scalar:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.subtract(x, y)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 0, -1, -2, -3, -4], dtype=int32)>
```

Note that binary `-` operator can be used instead:

```
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x - y
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
```

Subtract operation between an array and a tensor of same shape:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([5, 4, 3, 2, 1])
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 4,  2,  0, -2, -4], dtype=int32)>
```

**Warning**: If one of the inputs (`x` or `y`) is a tensor and the other is a
non-tensor, the non-tensor input will adopt (or get casted to) the data type
of the tensor input. This can potentially cause unwanted overflow or underflow
conversion.

For example,

```
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2**8 + 1, 2**8 + 2]
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2,), dtype=int8, numpy=array([0, 0], dtype=int8)>
```

When subtracting two input values of different shapes, <a href="../tf/math/subtract.md"><code>tf.subtract</code></a> follows the
[general broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
. The two input array shapes are compared element-wise. Starting with the
trailing dimensions, the two dimensions either have to be equal or one of them
needs to be `1`.

For example,

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(2, 1, 3)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 3), dtype=float64, numpy=
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],
       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]])>
```

Example with inputs of different dimensions:

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(1, 6)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 6), dtype=float64, numpy=
array([[[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `x`.
</td>
</tr>

</table>



<h3 id="__rtruediv__"><code>__rtruediv__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rtruediv__(
    x
)
</code></pre>

Divides x / y elementwise (using Python 3 division operator semantics).

NOTE: Prefer using the Tensor operator or tf.divide which obey Python
division operator semantics.

This function forces Python 3 division operator semantics where all integer
arguments are cast to floating types first.   This op is generated by normal
`x / y` division in Python 3 and in Python 2.7 with
`from __future__ import division`.  If you want integer division that rounds
down, use `x // y` or `tf.math.floordiv`.

`x` and `y` must have the same numeric type.  If the inputs are floating
point, the output will have the same type.  If the inputs are integral, the
inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
and `int64` (matching the behavior of Numpy).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
`Tensor` numerator of numeric type.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
`Tensor` denominator of numeric type.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`x / y` evaluated in floating point.
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
If `x` and `y` have different dtypes.
</td>
</tr>
</table>



<h3 id="__rxor__"><code>__rxor__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rxor__(
    x
)
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    y
)
</code></pre>

Returns x - y element-wise.

*NOTE*: <a href="../tf/math/subtract.md"><code>tf.subtract</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Both input and output have a range `(-inf, inf)`.

Example usages below.

Subtract operation between an array and a scalar:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.subtract(x, y)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 0, -1, -2, -3, -4], dtype=int32)>
```

Note that binary `-` operator can be used instead:

```
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x - y
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
```

Subtract operation between an array and a tensor of same shape:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([5, 4, 3, 2, 1])
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 4,  2,  0, -2, -4], dtype=int32)>
```

**Warning**: If one of the inputs (`x` or `y`) is a tensor and the other is a
non-tensor, the non-tensor input will adopt (or get casted to) the data type
of the tensor input. This can potentially cause unwanted overflow or underflow
conversion.

For example,

```
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2**8 + 1, 2**8 + 2]
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2,), dtype=int8, numpy=array([0, 0], dtype=int8)>
```

When subtracting two input values of different shapes, <a href="../tf/math/subtract.md"><code>tf.subtract</code></a> follows the
[general broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
. The two input array shapes are compared element-wise. Starting with the
trailing dimensions, the two dimensions either have to be equal or one of them
needs to be `1`.

For example,

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(2, 1, 3)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 3), dtype=float64, numpy=
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],
       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]])>
```

Example with inputs of different dimensions:

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(1, 6)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 6), dtype=float64, numpy=
array([[[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `x`.
</td>
</tr>

</table>



<h3 id="__truediv__"><code>__truediv__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    y
)
</code></pre>

Divides x / y elementwise (using Python 3 division operator semantics).

NOTE: Prefer using the Tensor operator or tf.divide which obey Python
division operator semantics.

This function forces Python 3 division operator semantics where all integer
arguments are cast to floating types first.   This op is generated by normal
`x / y` division in Python 3 and in Python 2.7 with
`from __future__ import division`.  If you want integer division that rounds
down, use `x // y` or `tf.math.floordiv`.

`x` and `y` must have the same numeric type.  If the inputs are floating
point, the output will have the same type.  If the inputs are integral, the
inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
and `int64` (matching the behavior of Numpy).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
`Tensor` numerator of numeric type.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
`Tensor` denominator of numeric type.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`x / y` evaluated in floating point.
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
If `x` and `y` have different dtypes.
</td>
</tr>
</table>



<h3 id="__xor__"><code>__xor__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__xor__(
    y
)
</code></pre>








<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
OVERLOADABLE_OPERATORS<a id="OVERLOADABLE_OPERATORS"></a>
</td>
<td>
```
{
 '__abs__',
 '__add__',
 '__and__',
 '__div__',
 '__eq__',
 '__floordiv__',
 '__ge__',
 '__getitem__',
 '__gt__',
 '__invert__',
 '__le__',
 '__lt__',
 '__matmul__',
 '__mod__',
 '__mul__',
 '__ne__',
 '__neg__',
 '__or__',
 '__pow__',
 '__radd__',
 '__rand__',
 '__rdiv__',
 '__rfloordiv__',
 '__rmatmul__',
 '__rmod__',
 '__rmul__',
 '__ror__',
 '__rpow__',
 '__rsub__',
 '__rtruediv__',
 '__rxor__',
 '__sub__',
 '__truediv__',
 '__xor__'
}
```
</td>
</tr>
</table>

