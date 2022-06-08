description: Computes the binary focal crossentropy loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.binary_focal_crossentropy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.metrics.binary_focal_crossentropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/losses.py#L1971-L2034">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the binary focal crossentropy loss.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.losses.binary_focal_crossentropy`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.losses.binary_focal_crossentropy`, `tf.compat.v1.keras.metrics.binary_focal_crossentropy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.binary_focal_crossentropy(
    y_true, y_pred, gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1
)
</code></pre>



<!-- Placeholder for "Used in" -->

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a focal factor to down-weight easy examples and focus more on
hard examples. By default, the focal tensor is computed as follows:

`focal_factor = (1 - output)**gamma` for class 1
`focal_factor = output**gamma` for class 0
where `gamma` is a focusing parameter. When `gamma` = 0, this function is
equivalent to the binary crossentropy loss.

#### Standalone usage:



```
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, gamma=2)
>>> assert loss.shape == (2,)
>>> loss.numpy()
array([0.330, 0.206], dtype=float32)
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
Ground truth values, of shape `(batch_size, d0, .. dN)`.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values, of shape `(batch_size, d0, .. dN)`.
</td>
</tr><tr>
<td>
`gamma`
</td>
<td>
A focusing parameter, default is `2.0` as mentioned in the reference.
</td>
</tr><tr>
<td>
`from_logits`
</td>
<td>
Whether `y_pred` is expected to be a logits tensor. By default,
we assume that `y_pred` encodes a probability distribution.
</td>
</tr><tr>
<td>
`label_smoothing`
</td>
<td>
Float in `[0, 1]`. If higher than 0 then smooth the labels
by squeezing them towards `0.5`, i.e., using `1. - 0.5 * label_smoothing`
for the target class and `0.5 * label_smoothing` for the non-target class.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
The axis along which the mean is computed. Defaults to `-1`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Binary focal crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
</td>
</tr>

</table>

