description: Optimizer that implements the FTRL algorithm.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.experimental.Ftrl" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_variable"/>
<meta itemprop="property" content="add_variable_from_reference"/>
<meta itemprop="property" content="aggregate_gradients"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_gradients"/>
<meta itemprop="property" content="finalize_variable_values"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="update_step"/>
</div>

# tf.keras.optimizers.experimental.Ftrl

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/ftrl.py#L25-L226">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Optimizer that implements the FTRL algorithm.

Inherits From: [`Optimizer`](../../../../tf/keras/optimizers/experimental/Optimizer.md), [`Module`](../../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.experimental.Ftrl(
    learning_rate=0.001,
    learning_rate_power=-0.5,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    l2_shrinkage_regularization_strength=0.0,
    beta=0.0,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name=&#x27;Ftrl&#x27;,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

"Follow The Regularized Leader" (FTRL) is an optimization algorithm developed
at Google for click-through rate prediction in the early 2010s. It is most
suitable for shallow models with large and sparse feature spaces.
The algorithm is described by
[McMahan et al., 2013](https://research.google.com/pubs/archive/41159.pdf).
The Keras version has support for both online L2 regularization
(the L2 regularization described in the paper
above) and shrinkage-type L2 regularization
(which is the addition of an L2 penalty to the loss function).

#### Initialization:



```python
n = 0
sigma = 0
z = 0
```

Update rule for one variable `w`:

```python
prev_n = n
n = n + g ** 2
sigma = (n ** -lr_power - prev_n ** -lr_power) / lr
z = z + g - sigma * w
if abs(z) < lambda_1:
  w = 0
else:
  w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / alpha + lambda_2)
```

#### Notation:



- `lr` is the learning rate
- `g` is the gradient for the variable
- `lambda_1` is the L1 regularization strength
- `lambda_2` is the L2 regularization strength
- `lr_power` is the power to scale n.

Check the documentation for the `l2_shrinkage_regularization_strength`
parameter for more details when shrinkage is enabled, in which case gradient
is replaced with a gradient with shrinkage.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate`
</td>
<td>
A `Tensor`, floating point value, a schedule that is a
<a href="../../../../tf/keras/optimizers/schedules/LearningRateSchedule.md"><code>tf.keras.optimizers.schedules.LearningRateSchedule</code></a>, or a callable that
takes no arguments and returns the actual value to use. The learning rate.
Defaults to 0.001.
</td>
</tr><tr>
<td>
`learning_rate_power`
</td>
<td>
A float value, must be less or equal to zero. Controls
how the learning rate decreases during training. Use zero for a fixed
learning rate.
</td>
</tr><tr>
<td>
`initial_accumulator_value`
</td>
<td>
The starting value for accumulators. Only zero or
positive values are allowed.
</td>
</tr><tr>
<td>
`l1_regularization_strength`
</td>
<td>
A float value, must be greater than or equal to
zero. Defaults to 0.0.
</td>
</tr><tr>
<td>
`l2_regularization_strength`
</td>
<td>
A float value, must be greater than or equal to
zero. Defaults to 0.0.
</td>
</tr><tr>
<td>
`l2_shrinkage_regularization_strength`
</td>
<td>
A float value, must be greater than or
equal to zero. This differs from L2 above in that the L2 above is a
stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
When input is sparse shrinkage will only happen on the active weights.
</td>
</tr><tr>
<td>
`beta`
</td>
<td>
A float value, representing the beta value from the paper. Defaults to
0.0.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String. The name to use
for momentum accumulator weights created by
the optimizer.
</td>
</tr><tr>
<td>
`clipnorm`
</td>
<td>
Float. If set, the gradient of each weight is individually
clipped so that its norm is no higher than this value.
</td>
</tr><tr>
<td>
`clipvalue`
</td>
<td>
Float. If set, the gradient of each weight is clipped to be no
higher than this value.
</td>
</tr><tr>
<td>
`global_clipnorm`
</td>
<td>
Float. If set, the gradient of all weights is clipped so
that their global norm is no higher than this value.
</td>
</tr><tr>
<td>
`use_ema`
</td>
<td>
Boolean, defaults to False. If True, exponential moving average
(EMA) is applied. EMA consists of computing an exponential moving
average of the weights of the model (as the weight values change after
each training batch), and periodically overwriting the weights with
their moving average.
</td>
</tr><tr>
<td>
`ema_momentum`
</td>
<td>
Float, defaults to 0.99. Only used if `use_ema=True`. This is
the momentum to use when computing the EMA of the model's weights:
`new_average = ema_momentum * old_average + (1 - ema_momentum) *
current_variable_value`.
</td>
</tr><tr>
<td>
`ema_overwrite_frequency`
</td>
<td>
Int or None, defaults to None. Only used if
`use_ema=True`. Every `ema_overwrite_frequency` steps of iterations, we
overwrite the model variable by its moving average. If None, the optimizer
 does not overwrite model variables in the middle of training, and you
need to explicitly overwrite the variables at the end of training
by calling `optimizer.finalize_variable_values()` (which updates the model
variables in-place). When using the built-in `fit()` training loop, this
happens automatically after the last epoch, and you don't need to do
anything.
</td>
</tr><tr>
<td>
`jit_compile`
</td>
<td>
Boolean, defaults to True. If True, the optimizer will use XLA
compilation. `jit_compile` cannot be True when training with
<a href="../../../../tf/distribute/experimental/ParameterServerStrategy.md"><code>tf.distribute.experimental.ParameterServerStrategy</code></a>. Additionally,
if no GPU device is found, this flag will be ignored.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments only used for backward compatibility.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`iterations`
</td>
<td>
The number of training steps this `optimizer` has run.

By default, iterations would be incremented by one every time
`apply_gradients()` is called.
</td>
</tr><tr>
<td>
`learning_rate`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="add_variable"><code>add_variable</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/optimizer.py#L319-L341">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_variable(
    shape, dtype=None, initializer=&#x27;zeros&#x27;, name=None
)
</code></pre>

Create an optimizer variable.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`shape`
</td>
<td>
A list of integers, a tuple of integers, or a 1-D Tensor of type
int32. Defaults to scalar if unspecified.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The DType of the optimizer variable to be created. Defaults to
<a href="../../../../tf/keras/backend/floatx.md"><code>tf.keras.backend.floatx</code></a> if unspecified.
</td>
</tr><tr>
<td>
`initializer`
</td>
<td>
string or callable. Initializer instance.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The name of the optimizer variable to be created.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An optimizer variable, in the format of tf.Variable.
</td>
</tr>

</table>



<h3 id="add_variable_from_reference"><code>add_variable_from_reference</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/optimizer.py#L760-L768">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_variable_from_reference(
    model_variable, variable_name, shape=None, initial_value=None
)
</code></pre>

Create an optimizer variable from model variable.

Create an optimizer variable based on the information of model variable.
For example, in SGD optimizer momemtum, for each model variable, a
corresponding momemtum variable is created of the same shape and dtype.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_variable`
</td>
<td>
tf.Variable. The corresponding model variable to the
optimizer variable to be created.
</td>
</tr><tr>
<td>
`variable_name`
</td>
<td>
String. The name prefix of the optimizer variable to be
created. The create variables name will follow the pattern
`{variable_name}/{model_variable.name}`, e.g., `momemtum/dense_1`.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
List or Tuple, defaults to None. The shape of the optimizer
variable to be created. If None, the created variable will have the
same shape as `model_variable`.
</td>
</tr><tr>
<td>
`initial_value`
</td>
<td>
A Tensor, or Python object convertible to a Tensor,
defaults to None. The initial value of the optimizer variable, if None,
the initial value will be default to 0.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An optimizer variable.
</td>
</tr>

</table>



<h3 id="aggregate_gradients"><code>aggregate_gradients</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/optimizer.py#L779-L791">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>aggregate_gradients(
    grads_and_vars
)
</code></pre>

Aggregate gradients on all devices.

By default we will perform reduce_sum of gradients across devices. Users can
implement their own aggregation logic by overriding this method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`grads_and_vars`
</td>
<td>
List of (gradient, variable) pairs.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of (gradient, variable) pairs.
</td>
</tr>

</table>



<h3 id="apply_gradients"><code>apply_gradients</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/optimizer.py#L793-L811">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_gradients(
    grads_and_vars, skip_gradients_aggregation=False
)
</code></pre>

Apply gradients to variables.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`grads_and_vars`
</td>
<td>
List of (gradient, variable) pairs.
</td>
</tr><tr>
<td>
`skip_gradients_aggregation`
</td>
<td>
If true, gradients aggregation will not be
performed inside optimizer. Usually this arg is set to True when you
write custom code aggregating gradients outside the optimizer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If `grads_and_vars` is malformed.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If called in a cross-replica context.
</td>
</tr>
</table>



<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/ftrl.py#L155-L177">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build(
    var_list
)
</code></pre>

Initialize optimizer variables.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var_list`
</td>
<td>
list of model variables to build Ftrl variables on.
</td>
</tr>
</table>



<h3 id="compute_gradients"><code>compute_gradients</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/optimizer.py#L152-L177">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_gradients(
    loss, var_list, tape=None
)
</code></pre>

Compute gradients of loss on trainable variables.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`loss`
</td>
<td>
`Tensor` or callable. If a callable, `loss` should take no arguments
and return the value to minimize.
</td>
</tr><tr>
<td>
`var_list`
</td>
<td>
list or tuple of `Variable` objects to update to minimize
`loss`.
</td>
</tr><tr>
<td>
`tape`
</td>
<td>
(Optional) <a href="../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a>. If `loss` is provided as a `Tensor`,
the tape that computed the `loss` must be provided.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of (gradient, variable) pairs. Variable is always present, but
gradient can be `None`.
</td>
</tr>

</table>



<h3 id="finalize_variable_values"><code>finalize_variable_values</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/optimizer.py#L469-L481">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>finalize_variable_values(
    var_list
)
</code></pre>

Set the final value of model's trainable variables.

Sometimes there are some extra steps before ending the variable updates,
such as overriding the model variables with its average value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var_list`
</td>
<td>
list of model variables.
</td>
</tr>
</table>



<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/optimizer.py#L518-L535">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Creates an optimizer from its config.

This method is the reverse of `get_config`, capable of instantiating the
same optimizer from the config dictionary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary, typically the output of get_config.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An optimizer instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/ftrl.py#L207-L226">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config of the optimizer.

An optimizer config is a Python dictionary (serializable)
containing the configuration of an optimizer.
The same optimizer can be reinstantiated later
(without any saved state) from this configuration.

Subclass optimizer should override this method to include other
hyperparameters.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Python dictionary.
</td>
</tr>

</table>



<h3 id="minimize"><code>minimize</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/optimizer.py#L382-L401">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>minimize(
    loss, var_list, tape=None
)
</code></pre>

Minimize `loss` by updating `var_list`.

This method simply computes gradient using <a href="../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a> and calls
`apply_gradients()`. If you want to process the gradient before applying
then call <a href="../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a> and `apply_gradients()` explicitly instead
of using this function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`loss`
</td>
<td>
`Tensor` or callable. If a callable, `loss` should take no arguments
and return the value to minimize.
</td>
</tr><tr>
<td>
`var_list`
</td>
<td>
list or tuple of `Variable` objects to update to minimize
`loss`.
</td>
</tr><tr>
<td>
`tape`
</td>
<td>
(Optional) <a href="../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None
</td>
</tr>

</table>



<h3 id="update_step"><code>update_step</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/ftrl.py#L179-L205">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_step(
    gradient, variable
)
</code></pre>

Update step given gradient and the associated model variable.




