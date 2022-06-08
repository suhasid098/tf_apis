description: Calculates how often predictions match binary labels.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.binary_accuracy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.metrics.binary_accuracy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/metrics/metrics.py#L3238-L3265">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Calculates how often predictions match binary labels.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.metrics.binary_accuracy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.binary_accuracy(
    y_true, y_pred, threshold=0.5
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Standalone usage:


```
>>> y_true = [[1], [1], [0], [0]]
>>> y_pred = [[1], [1], [0], [0]]
>>> m = tf.keras.metrics.binary_accuracy(y_true, y_pred)
>>> assert m.shape == (4,)
>>> m.numpy()
array([1., 1., 1., 1.], dtype=float32)
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
Ground truth values. shape = `[batch_size, d0, .. dN]`.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values. shape = `[batch_size, d0, .. dN]`.
</td>
</tr><tr>
<td>
`threshold`
</td>
<td>
(Optional) Float representing the threshold for deciding whether
prediction values are 1 or 0.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`
</td>
</tr>

</table>

