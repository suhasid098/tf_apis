description: Computes the minimum along segments of a tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.unsorted_segment_min" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.unsorted_segment_min

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes the minimum along segments of a tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.unsorted_segment_min`, `tf.compat.v1.unsorted_segment_min`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.unsorted_segment_min(
    data, segment_ids, num_segments, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to <a href="../../tf/math/unsorted_segment_sum.md"><code>tf.math.unsorted_segment_sum</code></a>,
Instead of computing the sum over segments, it computes the minimum such that:

\\(output_i = \min_{j...} data_[j...]\\) where min is over tuples `j...` such
that `segment_ids[j...] == i`.

If the minimum is empty for a given segment ID `i`, it outputs the largest
possible value for the specific numeric type,
`output[i] = numeric_limits<T>::max()`.

#### For example:



```
>>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
>>> tf.math.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
array([[1, 2, 2, 1],
       [5, 6, 7, 8]], dtype=int32)
```

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.

Caution: On CPU, values in `segment_ids` are always validated to be less than
`num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
result in safe but unspecified behavior, which may include ignoring
out-of-bound indices or outputting a tensor with a 0 stored in the first
dimension of its shape if `num_segments` is 0.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`segment_ids`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
A tensor whose shape is a prefix of `data.shape`.
The values must be less than `num_segments`.

Caution: The values are always validated to be in range on CPU, never validated
on GPU.
</td>
</tr><tr>
<td>
`num_segments`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
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
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `data`.
</td>
</tr>

</table>

