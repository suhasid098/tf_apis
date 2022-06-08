description: Calculate per-step mean Intersection-Over-Union (mIOU).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.metrics.mean_iou" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.metrics.mean_iou

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/metrics_impl.py">View source</a>



Calculate per-step mean Intersection-Over-Union (mIOU).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.metrics.mean_iou(
    labels,
    predictions,
    num_classes,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Mean Intersection-Over-Union is a common evaluation metric for
semantic image segmentation, which first computes the IOU for each
semantic class and then computes the average over classes.
IOU is defined as follows:
  IOU = true_positive / (true_positive + false_positive + false_negative).
The predictions are accumulated in a confusion matrix, weighted by `weights`,
and mIOU is then calculated from it.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the `mean_iou`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
A `Tensor` of ground truth labels with shape [batch size] and of
type `int32` or `int64`. The tensor will be flattened if its rank > 1.
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
A `Tensor` of prediction results for semantic labels, whose
shape is [batch size] and type `int32` or `int64`. The tensor will be
flattened if its rank > 1.
</td>
</tr><tr>
<td>
`num_classes`
</td>
<td>
The possible number of labels the prediction task can
have. This value must be provided, since a confusion matrix of
dimension = [num_classes, num_classes] will be allocated.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
Optional `Tensor` whose rank is either 0, or the same rank as
`labels`, and must be broadcastable to `labels` (i.e., all dimensions must
be either `1`, or the same as the corresponding `labels` dimension).
</td>
</tr><tr>
<td>
`metrics_collections`
</td>
<td>
An optional list of collections that `mean_iou`
should be added to.
</td>
</tr><tr>
<td>
`updates_collections`
</td>
<td>
An optional list of collections `update_op` should be
added to.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
An optional variable_scope name.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`mean_iou`
</td>
<td>
A `Tensor` representing the mean intersection-over-union.
</td>
</tr><tr>
<td>
`update_op`
</td>
<td>
An operation that increments the confusion matrix.
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
If `predictions` and `labels` have mismatched shapes, or if
`weights` is not `None` and its shape doesn't match `predictions`, or if
either `metrics_collections` or `updates_collections` are not a list or
tuple.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If eager execution is enabled.
</td>
</tr>
</table>

