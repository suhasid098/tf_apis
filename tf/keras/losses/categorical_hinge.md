description: Computes the categorical hinge loss between y_true and y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.losses.categorical_hinge" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.losses.categorical_hinge

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/losses.py#L1634-L1666">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the categorical hinge loss between `y_true` and `y_pred`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.losses.categorical_hinge`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.losses.categorical_hinge(
    y_true, y_pred
)
</code></pre>



<!-- Placeholder for "Used in" -->

`loss = maximum(neg - pos + 1, 0)`
where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

#### Standalone usage:



```
>>> y_true = np.random.randint(0, 3, size=(2,))
>>> y_true = tf.keras.utils.to_categorical(y_true, num_classes=3)
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.categorical_hinge(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> pos = np.sum(y_true * y_pred, axis=-1)
>>> neg = np.amax((1. - y_true) * y_pred, axis=-1)
>>> assert np.array_equal(loss.numpy(), np.maximum(0., neg - pos + 1.))
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
The ground truth values. `y_true` values are expected to be
either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor).
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Categorical hinge loss values.
</td>
</tr>

</table>

