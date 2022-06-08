description: Outputs the position of index in a permutation of [0, ..., max_index].

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.random.experimental.index_shuffle" />
<meta itemprop="path" content="Stable" />
</div>

# tf.random.experimental.index_shuffle

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/stateless_random_ops.py">View source</a>



Outputs the position of `index` in a permutation of [0, ..., max_index].

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.random.experimental.index_shuffle`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.random.experimental.index_shuffle(
    index, seed, max_index
)
</code></pre>



<!-- Placeholder for "Used in" -->

For each possible `seed` and `max_index` there is one pseudorandom permutation
of the sequence S=[0, ..., max_index]. Instead of materializing the full array
we can compute the new position of any single element in S. This can be useful
for very large `max_index`s.

The input `index` and output can be used as indices to shuffle a vector.
For example:

```
>>> vector = tf.constant(['e0', 'e1', 'e2', 'e3'])
>>> indices = tf.random.experimental.index_shuffle(tf.range(4), [5, 9], 3)
>>> shuffled_vector = tf.gather(vector, indices)
>>> print(shuffled_vector)
tf.Tensor([b'e2' b'e0' b'e1' b'e3'], shape=(4,), dtype=string)
```

More usefully, it can be used in a streaming (aka online) scenario such as
<a href="../../../tf/data.md"><code>tf.data</code></a>,  where each element of `vector` is processed individually and the
whole `vector` is never materialized in memory.

```
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.map(
...  lambda idx: tf.random.experimental.index_shuffle(idx, [5, 8], 9))
>>> print(list(dataset.as_numpy_iterator()))
[3, 8, 0, 1, 2, 7, 6, 9, 4, 5]
```

This operation is stateless (like other `tf.random.stateless_*` functions),
meaning the output is fully determined by the `seed` (other inputs being
equal).
Each `seed` choice corresponds to one permutation, so when calling this
function
multiple times for the same shuffling, please make sure to use the same
`seed`. For example:

```
>>> seed = [5, 9]
>>> idx0 = tf.random.experimental.index_shuffle(0, seed, 3)
>>> idx1 = tf.random.experimental.index_shuffle(1, seed, 3)
>>> idx2 = tf.random.experimental.index_shuffle(2, seed, 3)
>>> idx3 = tf.random.experimental.index_shuffle(3, seed, 3)
>>> shuffled_vector = tf.gather(vector, [idx0, idx1, idx2, idx3])
>>> print(shuffled_vector)
tf.Tensor([b'e2' b'e0' b'e1' b'e3'], shape=(4,), dtype=string)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`index`
</td>
<td>
An integer scalar tensor or vector with values in [0, `max_index`].
It can be seen as either a value `v` in the sequence `S`=[0, ...,
`max_index`] to be permutated, or as an index of an element `e` in a
shuffled vector.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A tensor of shape [2] or [n, 2] with dtype int32/uint32/int64/uint64.
The RNG seed. If the rank is unknown during graph building it must be 1 at
runtime.
</td>
</tr><tr>
<td>
`max_index`
</td>
<td>
A non-negative tensor with the same shape and dtype as `index`.
The upper bound (inclusive).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If all inputs were scalar (shape [2] for `seed`) the output will be a scalar
with the same dtype as `index`. The output can be seen as the new position
of `v` in `S`, or as the index of `e` in the vector before shuffling.
If one or multiple inputs were vectors (shape [n, 2] for `seed`) then the
output will be a vector of the same size which each element shuffled
independently. Scalar values are broadcasted in this case.
</td>
</tr>

</table>

