description: Computes the root mean squared error between the labels and predictions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.metrics.root_mean_squared_error" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.metrics.root_mean_squared_error

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/metrics_impl.py">View source</a>



Computes the root mean squared error between the labels and predictions.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.metrics.root_mean_squared_error(
    labels,
    predictions,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The `root_mean_squared_error` function creates two local variables,
`total` and `count` that are used to compute the root mean squared error.
This average is weighted by `weights`, and it is ultimately returned as
`root_mean_squared_error`: an idempotent operation that takes the square root
of the division of `total` by `count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`root_mean_squared_error`. Internally, a `squared_error` operation computes
the element-wise square of the difference between `predictions` and `labels`.
Then `update_op` increments `total` with the reduced sum of the product of
`weights` and `squared_error`, and it increments `count` with the reduced sum
of `weights`.

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
`root_mean_squared_error` should be added to.
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
`root_mean_squared_error`
</td>
<td>
A `Tensor` representing the current mean, the value
of `total` divided by `count`.
</td>
</tr><tr>
<td>
`update_op`
</td>
<td>
An operation that increments the `total` and `count` variables
appropriately and whose value matches `root_mean_squared_error`.
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

