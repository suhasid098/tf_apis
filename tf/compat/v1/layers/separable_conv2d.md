description: Functional interface for the depthwise separable 2D convolution layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.layers.separable_conv2d" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.layers.separable_conv2d

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/legacy_tf_layers/convolutional.py#L1312-L1476">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Functional interface for the depthwise separable 2D convolution layer.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.layers.separable_conv2d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding=&#x27;valid&#x27;,
    data_format=&#x27;channels_last&#x27;,
    dilation_rate=(1, 1),
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer=None,
    pointwise_initializer=None,
    bias_initializer=tf.compat.v1.zeros_initializer(),
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is a legacy api that is only compatible with eager execution and
<a href="../../../../tf/function.md"><code>tf.function</code></a> if you combine it with
<a href="../../../../tf/compat/v1/keras/utils/track_tf1_style_variables.md"><code>tf.compat.v1.keras.utils.track_tf1_style_variables</code></a>

Please refer to [tf.layers model mapping section of the migration guide]
(https://www.tensorflow.org/guide/migrate/model_mapping)
to learn how to use your TensorFlow v1 model in TF2 with Keras.

The corresponding TensorFlow v2 layer is
<a href="../../../../tf/keras/layers/SeparableConv2D.md"><code>tf.keras.layers.SeparableConv2D</code></a>.


#### Structural Mapping to Native TF2

None of the supported arguments have changed name.

Before:

```python
 y = tf.compat.v1.layers.separable_conv2d(x, filters=3, kernel_size=3)
```

After:

To migrate code using TF1 functional layers use the [Keras Functional API]
(https://www.tensorflow.org/guide/keras/functional):

```python
 x = tf.keras.Input((28, 28, 1))
 y = tf.keras.layers.SeparableConv2D(filters=3, kernels_size=3)(x)
 model = tf.keras.Model(x, y)
```


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

This layer performs a depthwise convolution that acts separately on
channels, followed by a pointwise convolution that mixes channels.
If `use_bias` is True and a bias initializer is provided,
it adds a bias vector to the output.
It then optionally applies an activation function to produce the final output.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor.
</td>
</tr><tr>
<td>
`filters`
</td>
<td>
Integer, the dimensionality of the output space (i.e. the number
of filters in the convolution).
</td>
</tr><tr>
<td>
`kernel_size`
</td>
<td>
A tuple or list of 2 integers specifying the spatial
dimensions of the filters. Can be a single integer to specify the same
value for all spatial dimensions.
</td>
</tr><tr>
<td>
`strides`
</td>
<td>
A tuple or list of 2 positive integers specifying the strides
of the convolution. Can be a single integer to specify the same value for
all spatial dimensions.
Specifying any `stride` value != 1 is incompatible with specifying
any `dilation_rate` value != 1.
</td>
</tr><tr>
<td>
`padding`
</td>
<td>
One of `"valid"` or `"same"` (case-insensitive).
`"valid"` means no padding. `"same"` results in padding evenly to
the left/right or up/down of the input such that output has the same
height/width dimension as the input.
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
A string, one of `channels_last` (default) or `channels_first`.
The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape
`(batch, height, width, channels)` while `channels_first` corresponds to
inputs with shape `(batch, channels, height, width)`.
</td>
</tr><tr>
<td>
`dilation_rate`
</td>
<td>
An integer or tuple/list of 2 integers, specifying
the dilation rate to use for dilated convolution.
Can be a single integer to specify the same value for
all spatial dimensions.
Currently, specifying any `dilation_rate` value != 1 is
incompatible with specifying any stride value != 1.
</td>
</tr><tr>
<td>
`depth_multiplier`
</td>
<td>
The number of depthwise convolution output channels for
each input channel. The total number of depthwise convolution output
channels will be equal to `num_filters_in * depth_multiplier`.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
Activation function. Set it to None to maintain a
linear activation.
</td>
</tr><tr>
<td>
`use_bias`
</td>
<td>
Boolean, whether the layer uses a bias.
</td>
</tr><tr>
<td>
`depthwise_initializer`
</td>
<td>
An initializer for the depthwise convolution kernel.
</td>
</tr><tr>
<td>
`pointwise_initializer`
</td>
<td>
An initializer for the pointwise convolution kernel.
</td>
</tr><tr>
<td>
`bias_initializer`
</td>
<td>
An initializer for the bias vector. If None, the default
initializer will be used.
</td>
</tr><tr>
<td>
`depthwise_regularizer`
</td>
<td>
Optional regularizer for the depthwise
convolution kernel.
</td>
</tr><tr>
<td>
`pointwise_regularizer`
</td>
<td>
Optional regularizer for the pointwise
convolution kernel.
</td>
</tr><tr>
<td>
`bias_regularizer`
</td>
<td>
Optional regularizer for the bias vector.
</td>
</tr><tr>
<td>
`activity_regularizer`
</td>
<td>
Optional regularizer function for the output.
</td>
</tr><tr>
<td>
`depthwise_constraint`
</td>
<td>
Optional projection function to be applied to the
depthwise kernel after being updated by an `Optimizer` (e.g. used for
norm constraints or value constraints for layer weights). The function
must take as input the unprojected variable and must return the
projected variable (which must have the same shape). Constraints are
not safe to use when doing asynchronous distributed training.
</td>
</tr><tr>
<td>
`pointwise_constraint`
</td>
<td>
Optional projection function to be applied to the
pointwise kernel after being updated by an `Optimizer`.
</td>
</tr><tr>
<td>
`bias_constraint`
</td>
<td>
Optional projection function to be applied to the
bias after being updated by an `Optimizer`.
</td>
</tr><tr>
<td>
`trainable`
</td>
<td>
Boolean, if `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a>).
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string, the name of the layer.
</td>
</tr><tr>
<td>
`reuse`
</td>
<td>
Boolean, whether to reuse the weights of a previous layer
by the same name.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Output tensor.
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
if eager execution is enabled.
</td>
</tr>
</table>


