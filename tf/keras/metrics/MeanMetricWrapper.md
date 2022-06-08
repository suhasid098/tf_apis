description: Wraps a stateless metric function with the Mean metric.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.MeanMetricWrapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.MeanMetricWrapper

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/base_metric.py#L582-L671">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Wraps a stateless metric function with the Mean metric.

Inherits From: [`Mean`](../../../tf/keras/metrics/Mean.md), [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.metrics.MeanMetricWrapper`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.MeanMetricWrapper(
    fn, name=None, dtype=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

You could use this class to quickly build a mean metric from a function. The
function needs to have the signature `fn(y_true, y_pred)` and return a
per-sample loss array. <a href="../../../tf/keras/metrics/Accuracy.md#result"><code>MeanMetricWrapper.result()</code></a> will return
the average metric value across all samples seen so far.

#### For example:



```python
def accuracy(y_true, y_pred):
  return tf.cast(tf.math.equal(y_true, y_pred), tf.float32)

accuracy_metric = tf.keras.metrics.MeanMetricWrapper(fn=accuracy)

keras_model.compile(..., metrics=accuracy_metric)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fn`
</td>
<td>
The metric function to wrap, with signature `fn(y_true, y_pred,
**kwargs)`.
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
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to pass on to `fn`.
</td>
</tr>
</table>



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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/base_metric.py#L238-L253">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/base_metric.py#L487-L498">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result()
</code></pre>

Computes and returns the scalar metric value tensor or a dict of scalars.

Result computation is an idempotent operation that simply calculates the
metric value using the state variables.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A scalar tensor, or a dictionary of scalar tensors.
</td>
</tr>

</table>



<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/base_metric.py#L616-L648">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates metric statistics.

For sparse categorical metrics, the shapes of `y_true` and `y_pred` are
different.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
Ground truth label values. shape = `[batch_size, d0, .. dN-1]` or
shape = `[batch_size, d0, .. dN-1, 1]`.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted probability values. shape = `[batch_size, d0, .. dN]`.
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional `sample_weight` acts as a
coefficient for the metric. If a scalar is provided, then the metric is
simply scaled by the given value. If `sample_weight` is a tensor of size
`[batch_size]`, then the metric for each sample of the batch is rescaled
by the corresponding element in the `sample_weight` vector. If the shape
of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
to this shape), then each metric element of `y_pred` is scaled by the
corresponding value of `sample_weight`. (Note on `dN-1`: all metric
functions reduce by 1 dimension, usually the last axis (-1)).
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





