description: DTensor specific optimizers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.dtensor.experimental.optimizers.Adam" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_variable"/>
<meta itemprop="property" content="add_variable_from_reference"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_gradients"/>
<meta itemprop="property" content="finalize_variable_values"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="update_step"/>
</div>

# tf.keras.dtensor.experimental.optimizers.Adam

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/dtensor/optimizers.py#L204-L222">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DTensor specific optimizers.

Inherits From: [`Adam`](../../../../../tf/keras/optimizers/experimental/Adam.md), [`Optimizer`](../../../../../tf/keras/optimizers/experimental/Optimizer.md), [`Module`](../../../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.dtensor.experimental.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    gradients_clip_option=None,
    ema_option=None,
    name=&#x27;Adam&#x27;,
    mesh=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The major changes for this class is that all the variable init logic will be
mesh/layout aware.

Optimizer that implements the Adam algorithm.

Adam optimization is a stochastic gradient descent method that is based on
adaptive estimation of first-order and second-order moments.

According to
[Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
the method is "*computationally
efficient, has little memory requirement, invariant to diagonal rescaling of
gradients, and is well suited for problems that are large in terms of
data/parameters*".

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate`
</td>
<td>
A <a href="../../../../../tf/Tensor.md"><code>tf.Tensor</code></a>, floating point value, a schedule that is a
<a href="../../../../../tf/keras/optimizers/schedules/LearningRateSchedule.md"><code>tf.keras.optimizers.schedules.LearningRateSchedule</code></a>, or a callable
that takes no arguments and returns the actual value to use. The
learning rate. Defaults to 0.001.
</td>
</tr><tr>
<td>
`beta_1`
</td>
<td>
A float value or a constant float tensor, or a callable
that takes no arguments and returns the actual value to use. The
exponential decay rate for the 1st moment estimates. Defaults to 0.9.
</td>
</tr><tr>
<td>
`beta_2`
</td>
<td>
A float value or a constant float tensor, or a callable
that takes no arguments and returns the actual value to use. The
exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
</td>
</tr><tr>
<td>
`epsilon`
</td>
<td>
A small constant for numerical stability. This epsilon is
"epsilon hat" in the Kingma and Ba paper (in the formula just before
Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
1e-7.
</td>
</tr><tr>
<td>
`amsgrad`
</td>
<td>
Boolean. Whether to apply AMSGrad variant of this algorithm from
the paper "On the Convergence of Adam and beyond". Defaults to `False`.
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
<a href="../../../../../tf/distribute/experimental/ParameterServerStrategy.md"><code>tf.distribute.experimental.ParameterServerStrategy</code></a>. Additionally,
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



#### Reference:

- [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
- [Reddi et al., 2018](
    https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.



#### Notes:



The default value of 1e-7 for epsilon might not be a good default in
general. For example, when training an Inception network on ImageNet a
current good choice is 1.0 or 0.1. Note that since Adam uses the
formulation just before Section 2.1 of the Kingma and Ba paper rather than
the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
hat" in the paper.

The sparse implementation of this algorithm (used when the gradient is an
IndexedSlices object, typically because of <a href="../../../../../tf/gather.md"><code>tf.gather</code></a> or an embedding
lookup in the forward pass) does apply momentum to variable slices even if
they were not used in the forward pass (meaning they have a gradient equal
to zero). Momentum decay (beta1) is also applied to the entire momentum
accumulator. This means that the sparse behavior is equivalent to the dense
behavior (in contrast to some momentum implementations which ignore momentum
unless a variable slice was actually used).



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
<a href="../../../../../tf/keras/backend/floatx.md"><code>tf.keras.backend.floatx</code></a> if unspecified.
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/dtensor/optimizers.py#L71-L105">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_variable_from_reference(
    model_variable, variable_name, initial_value=None
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
The corresponding model variable to the optimizer variable
to be created.
</td>
</tr><tr>
<td>
`variable_name`
</td>
<td>
The name prefix of the optimizer variable to be created.
The create variables name will follow the pattern
`{variable_name}/{model_variable.name}`, e.g., `momemtum/dense_1`.
</td>
</tr><tr>
<td>
`initial_value`
</td>
<td>
The initial value of the optimizer variable, if None, the
value will be default to 0.
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



<h3 id="apply_gradients"><code>apply_gradients</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/dtensor/optimizers.py#L117-L131">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_gradients(
    grads_and_vars
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
</tr>
</table>



<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/adam.py#L114-L141">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build(
    var_list
)
</code></pre>

Initialize optimizer variables.

Adam optimizer has 3 types of variables: momentums, velocities and
velocity_hat (only set when amsgrad is applied),

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var_list`
</td>
<td>
list of model variables to build Adam variables on.
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
(Optional) <a href="../../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a>. If `loss` is provided as a `Tensor`,
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/adam.py#L183-L193">View source</a>

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

This method simply computes gradient using <a href="../../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a> and calls
`apply_gradients()`. If you want to process the gradient before applying
then call <a href="../../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a> and `apply_gradients()` explicitly instead
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
(Optional) <a href="../../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a>.
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/optimizer_experimental/adam.py#L143-L181">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_step(
    gradient, variable
)
</code></pre>

Update step given gradient and the associated model variable.




