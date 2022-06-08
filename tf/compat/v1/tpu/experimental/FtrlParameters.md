description: Optimization parameters for Ftrl with TPU embeddings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.tpu.experimental.FtrlParameters" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.compat.v1.tpu.experimental.FtrlParameters

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding.py">View source</a>



Optimization parameters for Ftrl with TPU embeddings.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.tpu.experimental.FtrlParameters(
    learning_rate: float,
    learning_rate_power: float = -0.5,
    initial_accumulator_value: float = 0.1,
    l1_regularization_strength: float = 0.0,
    l2_regularization_strength: float = 0.0,
    use_gradient_accumulation: bool = True,
    clip_weight_min: Optional[float] = None,
    clip_weight_max: Optional[float] = None,
    weight_decay_factor: Optional[float] = None,
    multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
    multiply_linear_by_learning_rate: bool = False,
    beta: float = 0,
    allow_zero_accumulator: bool = False,
    clip_gradient_min: Optional[float] = None,
    clip_gradient_max: Optional[float] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
`optimization_parameters` argument to set the optimizer and its parameters.
See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
for more details.

```
estimator = tf.estimator.tpu.TPUEstimator(
    ...
    embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        ...
        optimization_parameters=tf.tpu.experimental.FtrlParameters(0.1),
        ...))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate`
</td>
<td>
a floating point value. The learning rate.
</td>
</tr><tr>
<td>
`learning_rate_power`
</td>
<td>
A float value, must be less or equal to zero.
Controls how the learning rate decreases during training. Use zero for a
fixed learning rate. See section 3.1 in the
[paper](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).
</td>
</tr><tr>
<td>
`initial_accumulator_value`
</td>
<td>
The starting value for accumulators. Only zero
or positive values are allowed.
</td>
</tr><tr>
<td>
`l1_regularization_strength`
</td>
<td>
A float value, must be greater than or equal
to zero.
</td>
</tr><tr>
<td>
`l2_regularization_strength`
</td>
<td>
A float value, must be greater than or equal
to zero.
</td>
</tr><tr>
<td>
`use_gradient_accumulation`
</td>
<td>
setting this to `False` makes embedding
gradients calculation less accurate but faster. Please see
`optimization_parameters.proto` for details. for details.
</td>
</tr><tr>
<td>
`clip_weight_min`
</td>
<td>
the minimum value to clip by; None means -infinity.
</td>
</tr><tr>
<td>
`clip_weight_max`
</td>
<td>
the maximum value to clip by; None means +infinity.
</td>
</tr><tr>
<td>
`weight_decay_factor`
</td>
<td>
amount of weight decay to apply; None means that the
weights are not decayed.
</td>
</tr><tr>
<td>
`multiply_weight_decay_factor_by_learning_rate`
</td>
<td>
if true,
`weight_decay_factor` is multiplied by the current learning rate.
</td>
</tr><tr>
<td>
`multiply_linear_by_learning_rate`
</td>
<td>
When true, multiplies the usages of the
linear slot in the weight update by the learning rate. This is useful
when ramping up learning rate from 0 (which would normally produce
NaNs).
</td>
</tr><tr>
<td>
`beta`
</td>
<td>
The beta parameter for FTRL.
</td>
</tr><tr>
<td>
`allow_zero_accumulator`
</td>
<td>
Changes the implementation of the square root to
allow for the case of initial_accumulator_value being zero. This will
cause a slight performance drop.
</td>
</tr><tr>
<td>
`clip_gradient_min`
</td>
<td>
the minimum value to clip by; None means -infinity.
Gradient accumulation must be set to true if this is set.
</td>
</tr><tr>
<td>
`clip_gradient_max`
</td>
<td>
the maximum value to clip by; None means +infinity.
Gradient accumulation must be set to true if this is set.
</td>
</tr>
</table>



