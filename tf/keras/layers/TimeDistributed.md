description: This wrapper allows to apply a layer to every temporal slice of an input.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.TimeDistributed" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.TimeDistributed

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/rnn/time_distributed.py#L30-L327">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



This wrapper allows to apply a layer to every temporal slice of an input.

Inherits From: [`Wrapper`](../../../tf/keras/layers/Wrapper.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.TimeDistributed`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.TimeDistributed(
    layer, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Every input should be at least 3D, and the dimension of index one of the
first input will be considered to be the temporal dimension.

Consider a batch of 32 video samples, where each sample is a 128x128 RGB image
with `channels_last` data format, across 10 timesteps.
The batch input shape is `(32, 10, 128, 128, 3)`.

You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
of the 10 timesteps, independently:

```
>>> inputs = tf.keras.Input(shape=(10, 128, 128, 3))
>>> conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3))
>>> outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
>>> outputs.shape
TensorShape([None, 10, 126, 126, 64])
```

Because `TimeDistributed` applies the same instance of `Conv2D` to each of the
timestamps, the same set of weights are used at each timestamp.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`layer`
</td>
<td>
a <a href="../../../tf/keras/layers/Layer.md"><code>tf.keras.layers.Layer</code></a> instance.
</td>
</tr>
</table>



#### Call arguments:


* <b>`inputs`</b>: Input tensor of shape (batch, time, ...) or nested tensors,
  and each of which has shape (batch, time, ...).
* <b>`training`</b>: Python boolean indicating whether the layer should behave in
  training mode or in inference mode. This argument is passed to the
  wrapped layer (only if the layer supports this argument).
* <b>`mask`</b>: Binary tensor of shape `(samples, timesteps)` indicating whether
  a given timestep should be masked. This argument is passed to the
  wrapped layer (only if the layer supports this argument).


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If not initialized with a <a href="../../../tf/keras/layers/Layer.md"><code>tf.keras.layers.Layer</code></a> instance.
</td>
</tr>
</table>



