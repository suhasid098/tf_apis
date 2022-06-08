description: Instantiates the ResNetRS50 architecture.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.applications.resnet_rs.ResNetRS50" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.applications.resnet_rs.ResNetRS50

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/applications/resnet_rs.py#L798-L824">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Instantiates the ResNetRS50 architecture.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.applications.ResNetRS50`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.applications.ResNetRS50`, `tf.compat.v1.keras.applications.resnet_rs.ResNetRS50`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.applications.resnet_rs.ResNetRS50(
    include_top=True,
    weights=&#x27;imagenet&#x27;,
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation=&#x27;softmax&#x27;,
    include_preprocessing=True
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Reference:


[Revisiting ResNets: Improved Training and Scaling Strategies](
https://arxiv.org/pdf/2103.07579.pdf)

For image classification use cases, see
[this page for detailed examples](
https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
https://keras.io/guides/transfer_learning/).

Note: each Keras Application expects a specific kind of input preprocessing.
For ResNetRs, by default input preprocessing is included as a part of the
model (as a `Rescaling` layer), and thus
<a href="../../../../tf/keras/applications/resnet_rs/preprocess_input.md"><code>tf.keras.applications.resnet_rs.preprocess_input</code></a> is actually a
pass-through function. In this use case, ResNetRS models expect their inputs
to be float tensors of pixels with values in the [0-255] range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to False.
With preprocessing disabled ResNetRS models expect their inputs to be float
tensors of pixels with values in the [-1, 1] range.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`depth`
</td>
<td>
Depth of ResNet network.
</td>
</tr><tr>
<td>
`input_shape`
</td>
<td>
optional shape tuple. It should have exactly 3 inputs
channels, and width and height should be no smaller than 32.
E.g. (200, 200, 3) would be one valid value.
</td>
</tr><tr>
<td>
`bn_momentum`
</td>
<td>
Momentum parameter for Batch Normalization layers.
</td>
</tr><tr>
<td>
`bn_epsilon`
</td>
<td>
Epsilon parameter for Batch Normalization layers.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
activation function.
</td>
</tr><tr>
<td>
`se_ratio`
</td>
<td>
Squeeze and Excitation layer ratio.
</td>
</tr><tr>
<td>
`dropout_rate`
</td>
<td>
dropout rate before final classifier layer.
</td>
</tr><tr>
<td>
`drop_connect_rate`
</td>
<td>
dropout rate at skip connections.
</td>
</tr><tr>
<td>
`include_top`
</td>
<td>
whether to include the fully-connected layer at the top of
the network.
</td>
</tr><tr>
<td>
`block_args`
</td>
<td>
list of dicts, parameters to construct block modules.
</td>
</tr><tr>
<td>
`model_name`
</td>
<td>
name of the model.
</td>
</tr><tr>
<td>
`pooling`
</td>
<td>
optional pooling mode for feature extraction when `include_top`
is `False`.
- `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
- `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
- `max` means that global max pooling will
    be applied.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
one of `None` (random initialization), `'imagenet'`
(pre-training on ImageNet), or the path to the weights file to be
loaded.  Note: one model can have multiple imagenet variants
depending on input shape it was trained with. For input_shape
224x224 pass `imagenet-i224` as argument. By default, highest input
shape weights are downloaded.
</td>
</tr><tr>
<td>
`input_tensor`
</td>
<td>
optional Keras tensor (i.e. output of <a href="../../../../tf/keras/Input.md"><code>layers.Input()</code></a>) to
use as image input for the model.
</td>
</tr><tr>
<td>
`classes`
</td>
<td>
optional number of classes to classify images into, only to be
specified if `include_top` is True, and if no `weights` argument is
specified.
</td>
</tr><tr>
<td>
`classifier_activation`
</td>
<td>
A `str` or callable. The activation function to
use on the "top" layer. Ignored unless `include_top=True`. Set
`classifier_activation=None` to return the logits of the "top" layer.
</td>
</tr><tr>
<td>
`include_preprocessing`
</td>
<td>
Boolean, whether to include the preprocessing layer
(`Rescaling`) at the bottom of the network. Defaults to `True`.
Note: Input image is normalized by ImageNet mean and standard deviation.
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

