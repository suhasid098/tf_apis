description: Computes the sum along sparse segments of a tensor divided by the sqrt(N).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.sparse.segment_sqrt_n" />
<meta itemprop="path" content="Stable" />
</div>

# tf.sparse.segment_sqrt_n

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Computes the sum along sparse segments of a tensor divided by the sqrt(N).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.sparse.segment_sqrt_n(
    data, indices, segment_ids, num_segments=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Read [the section on
segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
for an explanation of segments.

Like <a href="../../tf/sparse/segment_mean.md"><code>tf.sparse.segment_mean</code></a>, but instead of dividing by the size of the
segment, `N`, divide by `sqrt(N)` instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
A `Tensor` with data that will be assembled in the output.
</td>
</tr><tr>
<td>
`indices`
</td>
<td>
A 1-D `Tensor` with indices into `data`. Has same rank as
`segment_ids`.
</td>
</tr><tr>
<td>
`segment_ids`
</td>
<td>
A 1-D `Tensor` with indices into the output `Tensor`. Values
should be sorted and can be repeated.
</td>
</tr><tr>
<td>
`num_segments`
</td>
<td>
An optional int32 scalar. Indicates the size of the output
`Tensor`.
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
A `tensor` of the shape as data, except for dimension 0 which
has size `k`, the number of segments specified via `num_segments` or
inferred for the last element in `segments_ids`.
</td>
</tr>

</table>

