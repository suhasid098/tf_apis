description: A queue that produces elements sorted by the first component value.
robots: noindex

# tf.raw_ops.PriorityQueueV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



A queue that produces elements sorted by the first component value.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.PriorityQueueV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.PriorityQueueV2(
    shapes,
    component_types=[],
    capacity=-1,
    container=&#x27;&#x27;,
    shared_name=&#x27;&#x27;,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note that the PriorityQueue requires the first component of any element
to be a scalar int64, in addition to the other elements declared by
component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
entry in their input (resp. output) lists.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shapes`
</td>
<td>
A list of shapes (each a <a href="../../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `ints`).
The shape of each component in a value. The length of this attr must
be either 0 or the same as the length of component_types. If the length of
this attr is 0, the shapes of queue elements are not constrained, and
only one element may be dequeued at a time.
</td>
</tr><tr>
<td>
`component_types`
</td>
<td>
An optional list of `tf.DTypes`. Defaults to `[]`.
The type of each component in a value.
</td>
</tr><tr>
<td>
`capacity`
</td>
<td>
An optional `int`. Defaults to `-1`.
The upper bound on the number of elements in this queue.
Negative numbers mean no limit.
</td>
</tr><tr>
<td>
`container`
</td>
<td>
An optional `string`. Defaults to `""`.
If non-empty, this queue is placed in the given container.
Otherwise, a default container is used.
</td>
</tr><tr>
<td>
`shared_name`
</td>
<td>
An optional `string`. Defaults to `""`.
If non-empty, this queue will be shared under the given name
across multiple sessions.
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
A `Tensor` of type `resource`.
</td>
</tr>

</table>

