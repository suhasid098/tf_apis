description: Gather slices from params into a Tensor with shape specified by indices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.gather_nd" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.gather_nd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Gather slices from `params` into a Tensor with shape specified by `indices`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.manip.gather_nd`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.gather_nd(
    params, indices, name=None, batch_dims=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

`indices` is a `Tensor` of indices into `params`. The index vectors are
arranged along the last axis of `indices`.

This is similar to <a href="../../../tf/gather.md"><code>tf.gather</code></a>, in which `indices` defines slices into the
first dimension of `params`. In <a href="../../../tf/gather_nd.md"><code>tf.gather_nd</code></a>, `indices` defines slices into the
first `N` dimensions of `params`, where `N = indices.shape[-1]`.

Caution: On CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, a 0 is stored in the
corresponding output value.

## Gathering scalars

In the simplest case the vectors in `indices` index the full rank of `params`:

```
>>> tf.gather_nd(
...     indices=[[0, 0],
...              [1, 1]],
...     params = [['a', 'b'],
...               ['c', 'd']]).numpy()
array([b'a', b'd'], dtype=object)
```

In this case the result has 1-axis fewer than `indices`, and each index vector
is replaced by the scalar indexed from `params`.

In this case the shape relationship is:

```
index_depth = indices.shape[-1]
assert index_depth == params.shape.rank
result_shape = indices.shape[:-1]
```

If `indices` has a rank of `K`, it is helpful to think `indices` as a
(K-1)-dimensional tensor of indices into `params`.

## Gathering slices

If the index vectors do not index the full rank of `params` then each location
in the result contains a slice of params. This example collects rows from a
matrix:

```
>>> tf.gather_nd(
...     indices = [[1],
...                [0]],
...     params = [['a', 'b', 'c'],
...               ['d', 'e', 'f']]).numpy()
array([[b'd', b'e', b'f'],
       [b'a', b'b', b'c']], dtype=object)
```

Here `indices` contains `[2]` index vectors, each with a length of `1`.
The index vectors each refer to rows of the `params` matrix. Each
row has a shape of `[3]` so the output shape is `[2, 3]`.

In this case, the relationship between the shapes is:

```
index_depth = indices.shape[-1]
outer_shape = indices.shape[:-1]
assert index_depth <= params.shape.rank
inner_shape = params.shape[index_depth:]
output_shape = outer_shape + inner_shape
```

It is helpful to think of the results in this case as tensors-of-tensors.
The shape of the outer tensor is set by the leading dimensions of `indices`.
While the shape of the inner tensors is the shape of a single slice.

## Batches

Additionally both `params` and `indices` can have `M` leading batch
dimensions that exactly match. In this case `batch_dims` must be set to `M`.

For example, to collect one row from each of a batch of matrices you could
set the leading elements of the index vectors to be their location in the
batch:

```
>>> tf.gather_nd(
...     indices = [[0, 1],
...                [1, 0],
...                [2, 4],
...                [3, 2],
...                [4, 1]],
...     params=tf.zeros([5, 7, 3])).shape.as_list()
[5, 3]
```

The `batch_dims` argument lets you omit those leading location dimensions
from the index:

```
>>> tf.gather_nd(
...     batch_dims=1,
...     indices = [[1],
...                [0],
...                [4],
...                [2],
...                [1]],
...     params=tf.zeros([5, 7, 3])).shape.as_list()
[5, 3]
```

This is equivalent to caling a separate `gather_nd` for each location in the
batch dimensions.


```
>>> params=tf.zeros([5, 7, 3])
>>> indices=tf.zeros([5, 1])
>>> batch_dims = 1
>>>
>>> index_depth = indices.shape[-1]
>>> batch_shape = indices.shape[:batch_dims]
>>> assert params.shape[:batch_dims] == batch_shape
>>> outer_shape = indices.shape[batch_dims:-1]
>>> assert index_depth <= params.shape.rank
>>> inner_shape = params.shape[batch_dims + index_depth:]
>>> output_shape = batch_shape + outer_shape + inner_shape
>>> output_shape.as_list()
[5, 3]
```

### More examples

Indexing into a 3-tensor:

```
>>> tf.gather_nd(
...     indices = [[1]],
...     params = [[['a0', 'b0'], ['c0', 'd0']],
...               [['a1', 'b1'], ['c1', 'd1']]]).numpy()
array([[[b'a1', b'b1'],
        [b'c1', b'd1']]], dtype=object)
```



```
>>> tf.gather_nd(
...     indices = [[0, 1], [1, 0]],
...     params = [[['a0', 'b0'], ['c0', 'd0']],
...               [['a1', 'b1'], ['c1', 'd1']]]).numpy()
array([[b'c0', b'd0'],
       [b'a1', b'b1']], dtype=object)
```


```
>>> tf.gather_nd(
...     indices = [[0, 0, 1], [1, 0, 1]],
...     params = [[['a0', 'b0'], ['c0', 'd0']],
...               [['a1', 'b1'], ['c1', 'd1']]]).numpy()
array([b'b0', b'b1'], dtype=object)
```

The examples below are for the case when only indices have leading extra
dimensions. If both 'params' and 'indices' have leading batch dimensions, use
the 'batch_dims' parameter to run gather_nd in batch mode.

Batched indexing into a matrix:

```
>>> tf.gather_nd(
...     indices = [[[0, 0]], [[0, 1]]],
...     params = [['a', 'b'], ['c', 'd']]).numpy()
array([[b'a'],
       [b'b']], dtype=object)
```



Batched slice indexing into a matrix:

```
>>> tf.gather_nd(
...     indices = [[[1]], [[0]]],
...     params = [['a', 'b'], ['c', 'd']]).numpy()
array([[[b'c', b'd']],
       [[b'a', b'b']]], dtype=object)
```


Batched indexing into a 3-tensor:

```
>>> tf.gather_nd(
...     indices = [[[1]], [[0]]],
...     params = [[['a0', 'b0'], ['c0', 'd0']],
...               [['a1', 'b1'], ['c1', 'd1']]]).numpy()
array([[[[b'a1', b'b1'],
         [b'c1', b'd1']]],
       [[[b'a0', b'b0'],
         [b'c0', b'd0']]]], dtype=object)
```


```
>>> tf.gather_nd(
...     indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
...     params = [[['a0', 'b0'], ['c0', 'd0']],
...               [['a1', 'b1'], ['c1', 'd1']]]).numpy()
array([[[b'c0', b'd0'],
        [b'a1', b'b1']],
       [[b'a0', b'b0'],
        [b'c1', b'd1']]], dtype=object)
```

```
>>> tf.gather_nd(
...     indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]],
...     params = [[['a0', 'b0'], ['c0', 'd0']],
...               [['a1', 'b1'], ['c1', 'd1']]]).numpy()
array([[b'b0', b'b1'],
       [b'd0', b'c1']], dtype=object)
```


Examples with batched 'params' and 'indices':

```
>>> tf.gather_nd(
...     batch_dims = 1,
...     indices = [[1],
...                [0]],
...     params = [[['a0', 'b0'],
...                ['c0', 'd0']],
...               [['a1', 'b1'],
...                ['c1', 'd1']]]).numpy()
array([[b'c0', b'd0'],
       [b'a1', b'b1']], dtype=object)
```


```
>>> tf.gather_nd(
...     batch_dims = 1,
...     indices = [[[1]], [[0]]],
...     params = [[['a0', 'b0'], ['c0', 'd0']],
...               [['a1', 'b1'], ['c1', 'd1']]]).numpy()
array([[[b'c0', b'd0']],
       [[b'a1', b'b1']]], dtype=object)
```

```
>>> tf.gather_nd(
...     batch_dims = 1,
...     indices = [[[1, 0]], [[0, 1]]],
...     params = [[['a0', 'b0'], ['c0', 'd0']],
...               [['a1', 'b1'], ['c1', 'd1']]]).numpy()
array([[b'c0'],
       [b'b1']], dtype=object)
```


See also <a href="../../../tf/gather.md"><code>tf.gather</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`params`
</td>
<td>
A `Tensor`. The tensor from which to gather values.
</td>
</tr><tr>
<td>
`indices`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
Index tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr><tr>
<td>
`batch_dims`
</td>
<td>
An integer or a scalar 'Tensor'. The number of batch dimensions.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `params`.
</td>
</tr>

</table>

