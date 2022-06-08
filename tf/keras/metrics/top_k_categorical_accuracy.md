description: Computes how often targets are in the top K predictions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.top_k_categorical_accuracy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.metrics.top_k_categorical_accuracy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/metrics.py#L3336-L3363">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes how often targets are in the top `K` predictions.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.metrics.top_k_categorical_accuracy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.top_k_categorical_accuracy(
    y_true, y_pred, k=5
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Standalone usage:


```
>>> y_true = [[0, 0, 1], [0, 1, 0]]
>>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
>>> m = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
>>> assert m.shape == (2,)
>>> m.numpy()
array([1., 1.], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

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
The prediction values.
</td>
</tr><tr>
<td>
`k`
</td>
<td>
(Optional) Number of top elements to look at for computing accuracy.
Defaults to 5.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Top K categorical accuracy value.
</td>
</tr>

</table>

