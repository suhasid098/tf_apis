description: Computes a weighted cross entropy.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.weighted_cross_entropy_with_logits" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.weighted_cross_entropy_with_logits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_impl.py">View source</a>



Computes a weighted cross entropy.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.weighted_cross_entropy_with_logits(
    labels, logits, pos_weight, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
allows one to trade off recall and precision by up- or down-weighting the
cost of a positive error relative to a negative error.

The usual cross-entropy cost is defined as:

    labels * -log(sigmoid(logits)) +
        (1 - labels) * -log(1 - sigmoid(logits))

A value `pos_weight > 1` decreases the false negative count, hence increasing
the recall.
Conversely setting `pos_weight < 1` decreases the false positive count and
increases the precision.
This can be seen from the fact that `pos_weight` is introduced as a
multiplicative coefficient for the positive labels term
in the loss expression:

    labels * -log(sigmoid(logits)) * pos_weight +
        (1 - labels) * -log(1 - sigmoid(logits))

For brevity, let `x = logits`, `z = labels`, `q = pos_weight`.
The loss is:

      qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
    = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
    = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
    = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
    = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
the implementation uses

    (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

`logits` and `labels` must have the same type and shape.

```
>>> labels = tf.constant([1., 0.5, 0.])
>>> logits = tf.constant([1.5, -0.1, -10.])
>>> tf.nn.weighted_cross_entropy_with_logits(
...     labels=labels, logits=logits, pos_weight=tf.constant(1.5)).numpy()
array([3.0211994e-01, 8.8049585e-01, 4.5776367e-05], dtype=float32)
>>> tf.nn.weighted_cross_entropy_with_logits(
...     labels=labels, logits=logits, pos_weight=tf.constant(0.5)).numpy()
array([1.00706644e-01, 5.08297503e-01, 4.57763672e-05], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
A `Tensor` of the same type and shape as `logits`, with values
between 0 and 1 inclusive.
</td>
</tr><tr>
<td>
`logits`
</td>
<td>
A `Tensor` of type `float32` or `float64`, any real numbers.
</td>
</tr><tr>
<td>
`pos_weight`
</td>
<td>
A coefficient to use on the positive examples, typically a
scalar but otherwise broadcastable to the shape of `logits`. Its value
should be non-negative.
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
A `Tensor` of the same shape as `logits` with the componentwise
weighted logistic losses.
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
If `logits` and `labels` do not have the same shape.
</td>
</tr>
</table>

