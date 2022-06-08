description: Instantiates the MobileNet architecture.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.applications.mobilenet.MobileNet" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.applications.mobilenet.MobileNet

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/applications/mobilenet.py#L80-L311">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Instantiates the MobileNet architecture.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.applications.MobileNet`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.applications.MobileNet`, `tf.compat.v1.keras.applications.mobilenet.MobileNet`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.applications.mobilenet.MobileNet(
    input_shape=None,
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=True,
    weights=&#x27;imagenet&#x27;,
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation=&#x27;softmax&#x27;,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Reference:


- [MobileNets: Efficient Convolutional Neural Networks
   for Mobile Vision Applications](
    https://arxiv.org/abs/1704.04861)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples](
  https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
  https://keras.io/guides/transfer_learning/).

Note: each Keras Application expects a specific kind of input preprocessing.
For MobileNet, call <a href="../../../../tf/keras/applications/mobilenet/preprocess_input.md"><code>tf.keras.applications.mobilenet.preprocess_input</code></a>
on your inputs before passing them to the model.
<a href="../../../../tf/keras/applications/mobilenet/preprocess_input.md"><code>mobilenet.preprocess_input</code></a> will scale input pixels between -1 and 1.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Optional shape tuple, only to be specified if `include_top`
is False (otherwise the input shape has to be `(224, 224, 3)` (with
`channels_last` data format) or (3, 224, 224) (with `channels_first`
data format). It should have exactly 3 inputs channels, and width and
height should be no smaller than 32. E.g. `(200, 200, 3)` would be one
valid value. Default to `None`.
`input_shape` will be ignored if the `input_tensor` is provided.
</td>
</tr><tr>
<td>
`alpha`
</td>
<td>
Controls the width of the network. This is known as the width
multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
decreases the number of filters in each layer. - If `alpha` > 1.0,
proportionally increases the number of filters in each layer. - If
`alpha` = 1, default number of filters from the paper are used at each
layer. Default to 1.0.
</td>
</tr><tr>
<td>
`depth_multiplier`
</td>
<td>
Depth multiplier for depthwise convolution. This is
called the resolution multiplier in the MobileNet paper. Default to 1.0.
</td>
</tr><tr>
<td>
`dropout`
</td>
<td>
Dropout rate. Default to 0.001.
</td>
</tr><tr>
<td>
`include_top`
</td>
<td>
Boolean, whether to include the fully-connected layer at the
top of the network. Default to `True`.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
One of `None` (random initialization), 'imagenet' (pre-training
on ImageNet), or the path to the weights file to be loaded. Default to
`imagenet`.
</td>
</tr><tr>
<td>
`input_tensor`
</td>
<td>
Optional Keras tensor (i.e. output of <a href="../../../../tf/keras/Input.md"><code>layers.Input()</code></a>) to
use as image input for the model. `input_tensor` is useful for sharing
inputs between multiple different networks. Default to None.
</td>
</tr><tr>
<td>
`pooling`
</td>
<td>
Optional pooling mode for feature extraction when `include_top`
is `False`.
- `None` (default) means that the output of the model will be
    the 4D tensor output of the last convolutional block.
- `avg` means that global average pooling
    will be applied to the output of the
    last convolutional block, and thus
    the output of the model will be a 2D tensor.
- `max` means that global max pooling will be applied.
</td>
</tr><tr>
<td>
`classes`
</td>
<td>
Optional number of classes to classify images into, only to be
specified if `include_top` is True, and if no `weights` argument is
specified. Defaults to 1000.
</td>
</tr><tr>
<td>
`classifier_activation`
</td>
<td>
A `str` or callable. The activation function to use
on the "top" layer. Ignored unless `include_top=True`. Set
`classifier_activation=None` to return the logits of the "top" layer.
When loading pretrained weights, `classifier_activation` can only
be `None` or `"softmax"`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
For backwards compatibility only.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../../tf/keras/Model.md"><code>keras.Model</code></a> instance.
</td>
</tr>

</table>

