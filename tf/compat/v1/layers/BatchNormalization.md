description: Batch Normalization layer from (Ioffe et al., 2015).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.layers.BatchNormalization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_updates_for"/>
</div>

# tf.compat.v1.layers.BatchNormalization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/legacy_tf_layers/normalization.py#L31-L231">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Batch Normalization layer from (Ioffe et al., 2015).

Inherits From: [`BatchNormalization`](../../../../tf/compat/v1/keras/layers/BatchNormalization.md), [`Layer`](../../../../tf/compat/v1/layers/Layer.md), [`Layer`](../../../../tf/keras/layers/Layer.md), [`Module`](../../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.layers.BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=tf.compat.v1.zeros_initializer(),
    gamma_initializer=tf.compat.v1.ones_initializer(),
    moving_mean_initializer=tf.compat.v1.zeros_initializer(),
    moving_variance_initializer=tf.compat.v1.ones_initializer(),
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
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
<a href="../../../../tf/keras/layers/BatchNormalization.md"><code>tf.keras.layers.BatchNormalization</code></a>.


#### Structural Mapping to Native TF2

None of the supported arguments have changed name.

Before:

```python
 bn = tf.compat.v1.layers.BatchNormalization()
```

After:

```python
 bn = tf.keras.layers.BatchNormalization()
```

#### How to Map Arguments

TF1 Arg Name              | TF2 Arg Name              | Note
:------------------------ | :------------------------ | :---------------
`name`                    | `name`                    | Layer base class
`trainable`               | `trainable`               | Layer base class
`axis`                    | `axis`                    | -
`momentum`                | `momentum`                | -
`epsilon`                 | `epsilon`                 | -
`center`                  | `center`                  | -
`scale`                   | `scale`                   | -
`beta_initializer`        | `beta_initializer`        | -
`gamma_initializer`       | `gamma_initializer`       | -
`moving_mean_initializer` | `moving_mean_initializer` | -
`beta_regularizer`        | `beta_regularizer'        | -
`gamma_regularizer`       | `gamma_regularizer'       | -
`beta_constraint`         | `beta_constraint'         | -
`gamma_constraint`        | `gamma_constraint'        | -
`renorm`                  | Not supported             | -
`renorm_clipping`         | Not supported             | -
`renorm_momentum`         | Not supported             | -
`fused`                   | Not supported             | -
`virtual_batch_size`      | Not supported             | -
`adjustment`              | Not supported             | -



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Keras APIs handle BatchNormalization updates to the moving_mean and
moving_variance as part of their `fit()` and `evaluate()` loops. However, if a
custom training loop is used with an instance of `Model`, these updates need
to be explicitly included.  Here's a simple example of how it can be done:

```python
  # model is an instance of Model that contains BatchNormalization layer.
  update_ops = model.get_updates_for(None) + model.get_updates_for(features)
  train_op = optimizer.minimize(loss)
  train_op = tf.group([train_op, update_ops])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`axis`
</td>
<td>
An `int` or list of `int`, the axis or axes that should be normalized,
typically the features axis/axes. For instance, after a `Conv2D` layer
with `data_format="channels_first"`, set `axis=1`. If a list of axes is
provided, each axis in `axis` will be normalized
  simultaneously. Default is `-1` which uses the last axis. Note: when
    using multi-axis batch norm, the `beta`, `gamma`, `moving_mean`, and
    `moving_variance` variables are the same rank as the input Tensor,
    with dimension size 1 in all reduced (non-axis) dimensions).
</td>
</tr><tr>
<td>
`momentum`
</td>
<td>
Momentum for the moving average.
</td>
</tr><tr>
<td>
`epsilon`
</td>
<td>
Small float added to variance to avoid dividing by zero.
</td>
</tr><tr>
<td>
`center`
</td>
<td>
If True, add offset of `beta` to normalized tensor. If False, `beta`
is ignored.
</td>
</tr><tr>
<td>
`scale`
</td>
<td>
If True, multiply by `gamma`. If False, `gamma` is not used. When the
next layer is linear (also e.g. `nn.relu`), this can be disabled since the
scaling can be done by the next layer.
</td>
</tr><tr>
<td>
`beta_initializer`
</td>
<td>
Initializer for the beta weight.
</td>
</tr><tr>
<td>
`gamma_initializer`
</td>
<td>
Initializer for the gamma weight.
</td>
</tr><tr>
<td>
`moving_mean_initializer`
</td>
<td>
Initializer for the moving mean.
</td>
</tr><tr>
<td>
`moving_variance_initializer`
</td>
<td>
Initializer for the moving variance.
</td>
</tr><tr>
<td>
`beta_regularizer`
</td>
<td>
Optional regularizer for the beta weight.
</td>
</tr><tr>
<td>
`gamma_regularizer`
</td>
<td>
Optional regularizer for the gamma weight.
</td>
</tr><tr>
<td>
`beta_constraint`
</td>
<td>
An optional projection function to be applied to the `beta`
weight after being updated by an `Optimizer` (e.g. used to implement norm
constraints or value constraints for layer weights). The function must
take as input the unprojected variable and must return the projected
variable (which must have the same shape). Constraints are not safe to use
when doing asynchronous distributed training.
</td>
</tr><tr>
<td>
`gamma_constraint`
</td>
<td>
An optional projection function to be applied to the
`gamma` weight after being updated by an `Optimizer`.
</td>
</tr><tr>
<td>
`renorm`
</td>
<td>
Whether to use Batch Renormalization (Ioffe, 2017). This adds extra
variables during training. The inference is the same for either value of
this parameter.
</td>
</tr><tr>
<td>
`renorm_clipping`
</td>
<td>
A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
scalar `Tensors` used to clip the renorm correction. The correction `(r,
d)` is used as `corrected_value = normalized_value * r + d`, with `r`
clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
dmax are set to inf, 0, inf, respectively.
</td>
</tr><tr>
<td>
`renorm_momentum`
</td>
<td>
Momentum used to update the moving means and standard
deviations with renorm. Unlike `momentum`, this affects training and
should be neither too small (which would add noise) nor too large (which
would give stale estimates). Note that `momentum` is still applied to get
the means and variances for inference.
</td>
</tr><tr>
<td>
`fused`
</td>
<td>
if `None` or `True`, use a faster, fused implementation if possible.
If `False`, use the system recommended implementation.
</td>
</tr><tr>
<td>
`trainable`
</td>
<td>
Boolean, if `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
</td>
</tr><tr>
<td>
`virtual_batch_size`
</td>
<td>
An `int`. By default, `virtual_batch_size` is `None`,
which means batch normalization is performed across the whole batch. When
`virtual_batch_size` is not `None`, instead perform "Ghost Batch
Normalization", which creates virtual sub-batches which are each
normalized separately (with shared gamma, beta, and moving statistics).
Must divide the actual batch size during execution.
</td>
</tr><tr>
<td>
`adjustment`
</td>
<td>
A function taking the `Tensor` containing the (dynamic) shape of
the input tensor and returning a pair (scale, bias) to apply to the
normalized values (before gamma and beta), only during training. For
example, if axis==-1,
  `adjustment = lambda shape: (
    tf.random.uniform(shape[-1:], 0.93, 1.07),
    tf.random.uniform(shape[-1:], -0.1, 0.1))` will scale the normalized
      value by up to 7% up or down, then shift the result by up to 0.1
      (with independent scaling and bias for each feature but shared
      across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string, the name of the layer.
</td>
</tr>
</table>



#### References:

Batch Normalization - Accelerating Deep Network Training by Reducing
  Internal Covariate Shift:
  [Ioffe et al., 2015](http://proceedings.mlr.press/v37/ioffe15.html)
  ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
Batch Renormalization - Towards Reducing Minibatch Dependence in
  Batch-Normalized Models:
  [Ioffe,
    2017](http://papers.nips.cc/paper/6790-batch-renormalization-towards-reducing-minibatch-dependence-in-batch-normalized-models)
  ([pdf](http://papers.nips.cc/paper/6790-batch-renormalization-towards-reducing-minibatch-dependence-in-batch-normalized-models.pdf))







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`graph`
</td>
<td>

</td>
</tr><tr>
<td>
`scope_name`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="apply"><code>apply</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/legacy_tf_layers/base.py#L239-L240">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply(
    *args, **kwargs
)
</code></pre>




<h3 id="get_losses_for"><code>get_losses_for</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer_v1.py#L1341-L1358">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_losses_for(
    inputs
)
</code></pre>

Retrieves losses relevant to a specific set of inputs.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor or list/tuple of input tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of loss tensors of the layer that depend on `inputs`.
</td>
</tr>

</table>



<h3 id="get_updates_for"><code>get_updates_for</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer_v1.py#L1322-L1339">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_updates_for(
    inputs
)
</code></pre>

Retrieves updates relevant to a specific set of inputs.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor or list/tuple of input tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of update ops of the layer that depend on `inputs`.
</td>
</tr>

</table>





