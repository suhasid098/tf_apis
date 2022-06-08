description: Returns the indices of non-zero elements, or multiplexes x and y.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.where" />
<meta itemprop="path" content="Stable" />
</div>

# tf.where

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Returns the indices of non-zero elements, or multiplexes `x` and `y`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.where_v2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.where(
    condition, x=None, y=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation has two modes:

1. **Return the indices of non-zero elements** - When only
   `condition` is provided the result is an `int64` tensor where each row is
   the index of a non-zero element of `condition`. The result's shape
   is `[tf.math.count_nonzero(condition), tf.rank(condition)]`.
2. **Multiplex `x` and `y`** - When both `x` and `y` are provided the
   result has the shape of `x`, `y`, and `condition` broadcast together. The
   result is taken from `x` where `condition` is non-zero
   or `y` where `condition` is zero.

#### 1. Return the indices of non-zero elements

Note: In this mode `condition` can have a dtype of `bool` or any numeric
dtype.

If `x` and `y` are not provided (both are None):

<a href="../tf/where.md"><code>tf.where</code></a> will return the indices of `condition` that are non-zero,
in the form of a 2-D tensor with shape `[n, d]`, where `n` is the number of
non-zero elements in `condition` (`tf.count_nonzero(condition)`), and `d` is
the number of axes of `condition` (<a href="../tf/rank.md"><code>tf.rank(condition)</code></a>).

Indices are output in row-major order. The `condition` can have a `dtype` of
<a href="../tf.md#bool"><code>tf.bool</code></a>, or any numeric `dtype`.

Here `condition` is a 1-axis `bool` tensor with 2 `True` values. The result
has a shape of `[2,1]`

```
>>> tf.where([True, False, False, True]).numpy()
array([[0],
       [3]])
```

Here `condition` is a 2-axis integer tensor, with 3 non-zero values. The
result has a shape of `[3, 2]`.

```
>>> tf.where([[1, 0, 0], [1, 0, 1]]).numpy()
array([[0, 0],
       [1, 0],
       [1, 2]])
```

Here `condition` is a 3-axis float tensor, with 5 non-zero values. The output
shape is `[5, 3]`.

```
>>> float_tensor = [[[0.1, 0], [0, 2.2], [3.5, 1e6]],
...                 [[0,   0], [0,   0], [99,    0]]]
>>> tf.where(float_tensor).numpy()
array([[0, 0, 0],
       [0, 1, 1],
       [0, 2, 0],
       [0, 2, 1],
       [1, 2, 0]])
```

These indices are the same that <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> would use to
represent the condition tensor:

```
>>> sparse = tf.sparse.from_dense(float_tensor)
>>> sparse.indices.numpy()
array([[0, 0, 0],
       [0, 1, 1],
       [0, 2, 0],
       [0, 2, 1],
       [1, 2, 0]])
```

A complex number is considered non-zero if either the real or imaginary
component is non-zero:

```
>>> tf.where([complex(0.), complex(1.), 0+1j, 1+1j]).numpy()
array([[1],
       [2],
       [3]])
```

#### 2. Multiplex `x` and `y`

Note: In this mode `condition` must have a dtype of `bool`.

If `x` and `y` are also provided (both have non-None values) the `condition`
tensor acts as a mask that chooses whether the corresponding
element / row in the output should be taken from `x` (if the element in
`condition` is `True`) or `y` (if it is `False`).

The shape of the result is formed by
[broadcasting](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
together the shapes of `condition`, `x`, and `y`.

When all three inputs have the same size, each is handled element-wise.

```
>>> tf.where([True, False, False, True],
...          [1, 2, 3, 4],
...          [100, 200, 300, 400]).numpy()
array([  1, 200, 300,   4], dtype=int32)
```

There are two main rules for broadcasting:

1. If a tensor has fewer axes than the others, length-1 axes are added to the
   left of the shape.
2. Axes with length-1 are streched to match the coresponding axes of the other
   tensors.

A length-1 vector is streched to match the other vectors:

```
>>> tf.where([True, False, False, True], [1, 2, 3, 4], [100]).numpy()
array([  1, 100, 100,   4], dtype=int32)
```

A scalar is expanded to match the other arguments:

```
>>> tf.where([[True, False], [False, True]], [[1, 2], [3, 4]], 100).numpy()
array([[  1, 100], [100,   4]], dtype=int32)
>>> tf.where([[True, False], [False, True]], 1, 100).numpy()
array([[  1, 100], [100,   1]], dtype=int32)
```

A scalar `condition` returns the complete `x` or `y` tensor, with
broadcasting applied.

```
>>> tf.where(True, [1, 2, 3, 4], 100).numpy()
array([1, 2, 3, 4], dtype=int32)
>>> tf.where(False, [1, 2, 3, 4], 100).numpy()
array([100, 100, 100, 100], dtype=int32)
```

For a non-trivial example of broadcasting, here `condition` has a shape of
`[3]`, `x` has a shape of `[3,3]`, and `y` has a shape of `[3,1]`.
Broadcasting first expands the shape of `condition` to `[1,3]`. The final
broadcast shape is `[3,3]`. `condition` will select columns from `x` and `y`.
Since `y` only has one column, all columns from `y` will be identical.

```
>>> tf.where([True, False, True],
...          x=[[1, 2, 3],
...             [4, 5, 6],
...             [7, 8, 9]],
...          y=[[100],
...             [200],
...             [300]]
... ).numpy()
array([[ 1, 100, 3],
       [ 4, 200, 6],
       [ 7, 300, 9]], dtype=int32)
```

Note that if the gradient of either branch of the <a href="../tf/where.md"><code>tf.where</code></a> generates
a `NaN`, then the gradient of the entire <a href="../tf/where.md"><code>tf.where</code></a> will be `NaN`. This is
because the gradient calculation for <a href="../tf/where.md"><code>tf.where</code></a> combines the two branches, for
performance reasons.

A workaround is to use an inner <a href="../tf/where.md"><code>tf.where</code></a> to ensure the function has
no asymptote, and to avoid computing a value whose gradient is `NaN` by
replacing dangerous inputs with safe inputs.

Instead of this,

```
>>> x = tf.constant(0., dtype=tf.float32)
>>> with tf.GradientTape() as tape:
...   tape.watch(x)
...   y = tf.where(x < 1., 0., 1. / x)
>>> print(tape.gradient(y, x))
tf.Tensor(nan, shape=(), dtype=float32)
```

Although, the `1. / x` values are never used, its gradient is a `NaN` when
`x = 0`. Instead, we should guard that with another <a href="../tf/where.md"><code>tf.where</code></a>

```
>>> x = tf.constant(0., dtype=tf.float32)
>>> with tf.GradientTape() as tape:
...   tape.watch(x)
...   safe_x = tf.where(tf.equal(x, 0.), 1., x)
...   y = tf.where(x < 1., 0., 1. / safe_x)
>>> print(tape.gradient(y, x))
tf.Tensor(0.0, shape=(), dtype=float32)
```

#### See also:



* <a href="../tf/sparse.md"><code>tf.sparse</code></a> - The indices returned by the first form of <a href="../tf/where.md"><code>tf.where</code></a> can be
   useful in <a href="../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> objects.
* <a href="../tf/gather_nd.md"><code>tf.gather_nd</code></a>, <a href="../tf/scatter_nd.md"><code>tf.scatter_nd</code></a>, and related ops - Given the
  list of indices returned from <a href="../tf/where.md"><code>tf.where</code></a> the `scatter` and `gather` family
  of ops can be used fetch values or insert values at those indices.
* <a href="../tf/strings/length.md"><code>tf.strings.length</code></a> - <a href="../tf.md#string"><code>tf.string</code></a> is not an allowed dtype for the
  `condition`. Use the string length instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`condition`
</td>
<td>
A <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> of dtype bool, or any numeric dtype. `condition`
must have dtype `bool` when `x` and `y` are provided.
</td>
</tr><tr>
<td>
`x`
</td>
<td>
If provided, a Tensor which is of the same type as `y`, and has a shape
broadcastable with `condition` and `y`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
If provided, a Tensor which is of the same type as `x`, and has a shape
broadcastable with `condition` and `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name of the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `x` and `y` are provided:
  A `Tensor` with the same type as `x` and `y`, and shape that
  is broadcast from `condition`, `x`, and `y`.
Otherwise, a `Tensor` with shape `[tf.math.count_nonzero(condition),
tf.rank(condition)]`.
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
When exactly one of `x` or `y` is non-None, or the shapes
are not all broadcastable.
</td>
</tr>
</table>

