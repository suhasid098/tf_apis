description: Creates batches by randomly shuffling conditionally-enqueued tensors. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.train.maybe_shuffle_batch" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.train.maybe_shuffle_batch

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/input.py">View source</a>



Creates batches by randomly shuffling conditionally-enqueued tensors. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.train.maybe_shuffle_batch(
    tensors,
    batch_size,
    capacity,
    min_after_dequeue,
    keep_input,
    num_threads=1,
    seed=None,
    enqueue_many=False,
    shapes=None,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by <a href="../../../../tf/data.md"><code>tf.data</code></a>. Use `tf.data.Dataset.filter(...).shuffle(min_after_dequeue).batch(batch_size)`.

See docstring in `shuffle_batch` for more details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensors`
</td>
<td>
The list or dictionary of tensors to enqueue.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
The new batch size pulled from the queue.
</td>
</tr><tr>
<td>
`capacity`
</td>
<td>
An integer. The maximum number of elements in the queue.
</td>
</tr><tr>
<td>
`min_after_dequeue`
</td>
<td>
Minimum number elements in the queue after a
dequeue, used to ensure a level of mixing of elements.
</td>
</tr><tr>
<td>
`keep_input`
</td>
<td>
A `bool` Tensor.  This tensor controls whether the input is
added to the queue or not.  If it is a scalar and evaluates `True`, then
`tensors` are all added to the queue. If it is a vector and `enqueue_many`
is `True`, then each example is added to the queue only if the
corresponding value in `keep_input` is `True`. This tensor essentially
acts as a filtering mechanism.
</td>
</tr><tr>
<td>
`num_threads`
</td>
<td>
The number of threads enqueuing `tensor_list`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Seed for the random shuffling within the queue.
</td>
</tr><tr>
<td>
`enqueue_many`
</td>
<td>
Whether each tensor in `tensor_list` is a single example.
</td>
</tr><tr>
<td>
`shapes`
</td>
<td>
(Optional) The shapes for each example.  Defaults to the
inferred shapes for `tensor_list`.
</td>
</tr><tr>
<td>
`allow_smaller_final_batch`
</td>
<td>
(Optional) Boolean. If `True`, allow the final
batch to be smaller if there are insufficient items left in the queue.
</td>
</tr><tr>
<td>
`shared_name`
</td>
<td>
(Optional) If set, this queue will be shared under the given
name across multiple sessions.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(Optional) A name for the operations.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list or dictionary of tensors with the types as `tensors`.
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
If the `shapes` are not specified, and cannot be
inferred from the elements of `tensors`.
</td>
</tr>
</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the <a href="../../../../tf/data.md"><code>tf.data</code></a> API to ingest data under eager execution.


 </devsite-expandable></section>

