description: Computes the Intersection-Over-Union metric for specific target classes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.IoU" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.IoU

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/metrics.py#L2507-L2626">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the Intersection-Over-Union metric for specific target classes.

Inherits From: [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.metrics.IoU`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.IoU(
    num_classes: int,
    target_class_ids: Union[List[int], Tuple[int, ...]],
    name=None,
    dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

General definition and computation:

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

For an individual class, the IoU metric is defined as follows:

```
iou = true_positives / (true_positives + false_positives + false_negatives)
```

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

Note, this class first computes IoUs for all individual classes, then returns
the mean of IoUs for the classes that are specified by `target_class_ids`. If
`target_class_ids` has only one id value, the IoU of that specific class is
returned.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_classes`
</td>
<td>
The possible number of labels the prediction task can have.
A confusion matrix of dimension = [num_classes, num_classes] will be
allocated to accumulate predictions from which the metric is calculated.
</td>
</tr><tr>
<td>
`target_class_ids`
</td>
<td>
A tuple or list of target class ids for which the metric
is returned. To compute IoU for a specific class, a list (or tuple) of a
single id value should be provided.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(Optional) string name of the metric instance.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
(Optional) data type of the metric result.
</td>
</tr>
</table>



#### Standalone usage:



```
>>> # cm = [[1, 1],
>>> #        [1, 1]]
>>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
>>> # iou = true_positives / (sum_row + sum_col - true_positives))
>>> # iou = [0.33, 0.33]
>>> m = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
>>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
>>> m.result().numpy()
0.33333334
```

```
>>> m.reset_state()
>>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
...                sample_weight=[0.3, 0.3, 0.3, 0.1])
>>> # cm = [[0.3, 0.3],
>>> #        [0.3, 0.1]]
>>> # sum_row = [0.6, 0.4], sum_col = [0.6, 0.4], true_positives = [0.3, 0.1]
>>> # iou = [0.33, 0.14]
>>> m.result().numpy()
0.33333334
```

Usage with `compile()` API:

```python
model.compile(
  optimizer='sgd',
  loss='mse',
  metrics=[tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])])
```

## Methods

<h3 id="merge_state"><code>merge_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/base_metric.py#L275-L309">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_state(
    metrics
)
</code></pre>

Merges the state from one or more metrics.

This method can be used by distributed systems to merge the state computed
by different metric instances. Typically the state will be stored in the
form of the metric's weights. For example, a tf.keras.metrics.Mean metric
contains a list of two weight values: a total and a count. If there were two
instances of a tf.keras.metrics.Accuracy that each independently aggregated
partial state for an overall accuracy calculation, these two metric's states
could be combined as follows:

```
>>> m1 = tf.keras.metrics.Accuracy()
>>> _ = m1.update_state([[1], [2]], [[0], [2]])
```

```
>>> m2 = tf.keras.metrics.Accuracy()
>>> _ = m2.update_state([[3], [4]], [[3], [4]])
```

```
>>> m2.merge_state([m1])
>>> m2.result().numpy()
0.75
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`metrics`
</td>
<td>
an iterable of metrics. The metrics must have compatible state.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided iterable does not contain metrics matching the
metric's required specifications.
</td>
</tr>
</table>



<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/metrics.py#L2502-L2504">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/metrics.py#L2594-L2618">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result()
</code></pre>

Compute the intersection-over-union via the confusion matrix.


<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/metrics.py#L2464-L2500">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates the confusion matrix statistics.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
The ground truth values.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values.
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional weighting of each example. Defaults to 1. Can be a
`Tensor` whose rank is either 0, or the same rank as `y_true`, and must
be broadcastable to `y_true`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Update op.
</td>
</tr>

</table>





