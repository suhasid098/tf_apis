description: Computes the mean relative error by normalizing with the given values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.metrics.mean_relative_error" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.metrics.mean_relative_error

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/metrics_impl.py">View source</a>



Computes the mean relative error by normalizing with the given values.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.metrics.mean_relative_error(
    labels,
    predictions,
    normalizer,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The `mean_relative_error` function creates two local variables,
`total` and `count` that are used to compute the mean relative absolute error.
This average is weighted by `weights`, and it is ultimately returned as
`mean_relative_error`: an idempotent operation that simply divides `total` by
`count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`mean_reative_error`. Internally, a `relative_errors` operation divides the
absolute value of the differences between `predictions` and `labels` by the
`normalizer`. Then `update_op` increments `total` with the reduced sum of the
product of `weights` and `relative_errors`, and it increments `count` with the
reduced sum of `weights`.

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
A `Tensor` of the same shape as `predictions`.
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
A `Tensor` of arbitrary shape.
</td>
</tr><tr>
<td>
`normalizer`
</td>
<td>
A `Tensor` of the same shape as `predictions`.
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
An optional list of collections that
`mean_relative_error` should be added to.
</td>
</tr><tr>
<td>
`updates_collections`
</td>
<td>
An optional list of collections that `update_op` should
be added to.
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
`mean_relative_error`
</td>
<td>
A `Tensor` representing the current mean, the value of
`total` divided by `count`.
</td>
</tr><tr>
<td>
`update_op`
</td>
<td>
An operation that increments the `total` and `count` variables
appropriately and whose value matches `mean_relative_error`.
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

