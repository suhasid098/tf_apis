description: Instantiates the Densenet201 architecture.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.applications.densenet.DenseNet201" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.applications.densenet.DenseNet201

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/applications/densenet.py#L352-L363">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Instantiates the Densenet201 architecture.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.applications.DenseNet201`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.applications.DenseNet201`, `tf.compat.v1.keras.applications.densenet.DenseNet201`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.applications.densenet.DenseNet201(
    include_top=True,
    weights=&#x27;imagenet&#x27;,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation=&#x27;softmax&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Reference:


- [Densely Connected Convolutional Networks](
    https://arxiv.org/abs/1608.06993) (CVPR 2017)

Optionally loads weights pre-trained on ImageNet.
Note that the data format convention used by the model is
the one specified in your Keras config at `~/.keras/keras.json`.

Note: each Keras Application expects a specific kind of input preprocessing.
For DenseNet, call <a href="../../../../tf/keras/applications/densenet/preprocess_input.md"><code>tf.keras.applications.densenet.preprocess_input</code></a> on your
inputs before passing them to the model.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`include_top`
</td>
<td>
whether to include the fully-connected
layer at the top of the network.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
one of `None` (random initialization),
'imagenet' (pre-training on ImageNet),
or the path to the weights file to be loaded.
</td>
</tr><tr>
<td>
`input_tensor`
</td>
<td>
optional Keras tensor (i.e. output of <a href="../../../../tf/keras/Input.md"><code>layers.Input()</code></a>)
to use as image input for the model.
</td>
</tr><tr>
<td>
`input_shape`
</td>
<td>
optional shape tuple, only to be specified
if `include_top` is False (otherwise the input shape
has to be `(224, 224, 3)` (with `'channels_last'` data format)
or `(3, 224, 224)` (with `'channels_first'` data format).
It should have exactly 3 inputs channels,
and width and height should be no smaller than 32.
E.g. `(200, 200, 3)` would be one valid value.
</td>
</tr><tr>
<td>
`pooling`
</td>
<td>
Optional pooling mode for feature extraction
when `include_top` is `False`.
- `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional block.
- `avg` means that global average pooling
    will be applied to the output of the
    last convolutional block, and thus
    the output of the model will be a 2D tensor.
- `max` means that global max pooling will
    be applied.
</td>
</tr><tr>
<td>
`classes`
</td>
<td>
optional number of classes to classify images
into, only to be specified if `include_top` is True, and
if no `weights` argument is specified.
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
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Keras model instance.
</td>
</tr>

</table>

