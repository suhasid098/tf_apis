description: This is the class from which all layers inherit.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Layer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="add_loss"/>
<meta itemprop="property" content="add_metric"/>
<meta itemprop="property" content="add_weight"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="compute_output_signature"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="set_weights"/>
</div>

# tf.keras.layers.Layer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L125-L3203">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



This is the class from which all layers inherit.

Inherits From: [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.Layer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Layer(
    trainable=True, name=None, dtype=None, dynamic=False, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

A layer is a callable object that takes as input one or more tensors and
that outputs one or more tensors. It involves *computation*, defined
in the `call()` method, and a *state* (weight variables). State can be
created in various places, at the convenience of the subclass implementer:

* in `__init__()`;
* in the optional `build()` method, which is invoked by the first
  `__call__()` to the layer, and supplies the shape(s) of the input(s),
  which may not have been known at initialization time;
* in the first invocation of `call()`, with some caveats discussed
  below.

Users will just instantiate a layer and then treat it as a callable.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`trainable`
</td>
<td>
Boolean, whether the layer's variables should be trainable.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String name of the layer.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The dtype of the layer's computations and weights. Can also be a
<a href="../../../tf/keras/mixed_precision/Policy.md"><code>tf.keras.mixed_precision.Policy</code></a>, which allows the computation and weight
dtype to differ. Default of `None` means to use
<a href="../../../tf/keras/mixed_precision/global_policy.md"><code>tf.keras.mixed_precision.global_policy()</code></a>, which is a float32 policy
unless set to different value.
</td>
</tr><tr>
<td>
`dynamic`
</td>
<td>
Set this to `True` if your layer should only be run eagerly, and
should not be used to generate a static computation graph.
This would be the case for a Tree-RNN or a recursive network,
for example, or generally for any layer that manipulates tensors
using Python control flow. If `False`, we assume that the layer can
safely be used to generate a static computation graph.
</td>
</tr>
</table>


We recommend that descendants of `Layer` implement the following methods:

* `__init__()`: Defines custom layer attributes, and creates layer weights
  that do not depend on input shapes, using `add_weight()`, or other state.
* `build(self, input_shape)`: This method can be used to create weights that
  depend on the shape(s) of the input(s), using `add_weight()`, or other
  state. `__call__()` will automatically build the layer (if it has not been
  built yet) by calling `build()`.
* `call(self, inputs, *args, **kwargs)`: Called in `__call__` after making
  sure `build()` has been called. `call()` performs the logic of applying the
  layer to the `inputs`. The first invocation may additionally create state
  that could not be conveniently created in `build()`; see its docstring
  for details.
  Two reserved keyword arguments you can optionally use in `call()` are:
    - `training` (boolean, whether the call is in inference mode or training
      mode). See more details in [the layer/model subclassing guide](
      https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_training_argument_in_the_call_method)
    - `mask` (boolean tensor encoding masked timesteps in the input, used
      in RNN layers). See more details in [the layer/model subclassing guide](
      https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_mask_argument_in_the_call_method)
  A typical signature for this method is `call(self, inputs)`, and user could
  optionally add `training` and `mask` if the layer need them. `*args` and
  `**kwargs` is only useful for future extension when more input parameters
  are planned to be added.
* `get_config(self)`: Returns a dictionary containing the configuration used
  to initialize this layer. If the keys differ from the arguments
  in `__init__`, then override `from_config(self)` as well.
  This method is used when saving
  the layer or a model that contains this layer.

#### Examples:



Here's a basic example: a layer with two variables, `w` and `b`,
that returns `y = w . x + b`.
It shows how to implement `build()` and `call()`.
Variables set as attributes of a layer are tracked as weights
of the layers (in `layer.weights`).

```python
class SimpleDense(Layer):

  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):  # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'),
        trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True)

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.matmul(inputs, self.w) + self.b

# Instantiates the layer.
linear_layer = SimpleDense(4)

# This will also call `build(input_shape)` and create the weights.
y = linear_layer(tf.ones((2, 2)))
assert len(linear_layer.weights) == 2

# These weights are trainable, so they're listed in `trainable_weights`:
assert len(linear_layer.trainable_weights) == 2
```

Note that the method `add_weight()` offers a shortcut to create weights:

```python
class SimpleDense(Layer):

  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):
      self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
      self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

  def call(self, inputs):
      return tf.matmul(inputs, self.w) + self.b
```

Besides trainable weights, updated via backpropagation during training,
layers can also have non-trainable weights. These weights are meant to
be updated manually during `call()`. Here's a example layer that computes
the running sum of its inputs:

```python
class ComputeSum(Layer):

  def __init__(self, input_dim):
      super(ComputeSum, self).__init__()
      # Create a non-trainable weight.
      self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                               trainable=False)

  def call(self, inputs):
      self.total.assign_add(tf.reduce_sum(inputs, axis=0))
      return self.total

my_sum = ComputeSum(2)
x = tf.ones((2, 2))

y = my_sum(x)
print(y.numpy())  # [2. 2.]

y = my_sum(x)
print(y.numpy())  # [4. 4.]

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
```

For more information about creating layers, see the guide
[Making new Layers and Models via subclassing](
  https://www.tensorflow.org/guide/keras/custom_layers_and_models)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
The name of the layer (string).
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The dtype of the layer's weights.
</td>
</tr><tr>
<td>
`variable_dtype`
</td>
<td>
Alias of `dtype`.
</td>
</tr><tr>
<td>
`compute_dtype`
</td>
<td>
The dtype of the layer's computations. Layers automatically
cast inputs to this dtype which causes the computations and output to also
be in this dtype. When mixed precision is used with a
<a href="../../../tf/keras/mixed_precision/Policy.md"><code>tf.keras.mixed_precision.Policy</code></a>, this will be different than
`variable_dtype`.
</td>
</tr><tr>
<td>
`dtype_policy`
</td>
<td>
The layer's dtype policy. See the
<a href="../../../tf/keras/mixed_precision/Policy.md"><code>tf.keras.mixed_precision.Policy</code></a> documentation for details.
</td>
</tr><tr>
<td>
`trainable_weights`
</td>
<td>
List of variables to be included in backprop.
</td>
</tr><tr>
<td>
`non_trainable_weights`
</td>
<td>
List of variables that should not be
included in backprop.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
The concatenation of the lists trainable_weights and
non_trainable_weights (in this order).
</td>
</tr><tr>
<td>
`trainable`
</td>
<td>
Whether the layer should be trained (boolean), i.e. whether
its potentially-trainable weights should be returned as part of
`layer.trainable_weights`.
</td>
</tr><tr>
<td>
`input_spec`
</td>
<td>
Optional (list of) `InputSpec` object(s) specifying the
constraints on inputs that can be accepted by the layer.
</td>
</tr><tr>
<td>
`activity_regularizer`
</td>
<td>
Optional regularizer function for the output of this layer.
</td>
</tr><tr>
<td>
`dynamic`
</td>
<td>
Whether the layer is dynamic (eager-only); set in the constructor.
</td>
</tr><tr>
<td>
`input`
</td>
<td>
Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.
</td>
</tr><tr>
<td>
`losses`
</td>
<td>
List of losses added using the `add_loss()` API.

Variable regularization tensors are created when this property is accessed,
so it is eager safe: accessing `losses` under a <a href="../../../tf/GradientTape.md"><code>tf.GradientTape</code></a> will
propagate gradients back to the corresponding variables.

```
>>> class MyLayer(tf.keras.layers.Layer):
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
>>> l = MyLayer()
>>> l(np.ones((10, 1)))
>>> l.losses
[1.0]
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> x = tf.keras.layers.Dense(10)(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Activity regularization.
>>> len(model.losses)
0
>>> model.add_loss(tf.abs(tf.reduce_mean(x)))
>>> len(model.losses)
1
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
>>> x = d(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Weight regularization.
>>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
>>> model.losses
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
```
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
List of metrics added using the `add_metric()` API.


```
>>> input = tf.keras.layers.Input(shape=(3,))
>>> d = tf.keras.layers.Dense(2)
>>> output = d(input)
>>> d.add_metric(tf.reduce_max(output), name='max')
>>> d.add_metric(tf.reduce_min(output), name='min')
>>> [m.name for m in d.metrics]
['max', 'min']
```
</td>
</tr><tr>
<td>
`output`
</td>
<td>
Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.
</td>
</tr><tr>
<td>
`supports_masking`
</td>
<td>
Whether this layer supports computing a mask using `compute_mask`.
</td>
</tr>
</table>



## Methods

<h3 id="add_loss"><code>add_loss</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L1273-L1389">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_loss(
    losses, **kwargs
)
</code></pre>

Add loss tensor(s), potentially dependent on layer inputs.

Some losses (for instance, activity regularization losses) may be dependent
on the inputs passed when calling a layer. Hence, when reusing the same
layer on different inputs `a` and `b`, some entries in `layer.losses` may
be dependent on `a` and some on `b`. This method automatically keeps track
of dependencies.

This method can be used inside a subclassed layer or model's `call`
function, in which case `losses` should be a Tensor or list of Tensors.

#### Example:



```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any loss Tensors passed to this Model must
be symbolic and be able to be traced back to the model's `Input`s. These
losses become part of the model's topology and are tracked in `get_config`.

#### Example:



```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Activity regularization.
model.add_loss(tf.abs(tf.reduce_mean(x)))
```

If this is not the case for your loss (if, for example, your loss references
a `Variable` of one of the model's layers), you can wrap your loss in a
zero-argument lambda. These losses are not tracked as part of the model's
topology since they can't be serialized.

#### Example:



```python
inputs = tf.keras.Input(shape=(10,))
d = tf.keras.layers.Dense(10)
x = d(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Weight regularization.
model.add_loss(lambda: tf.reduce_mean(d.kernel))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`losses`
</td>
<td>
Loss tensor, or list/tuple of tensors. Rather than tensors, losses
may also be zero-argument callables which create a loss tensor.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Used for backwards compatibility only.
</td>
</tr>
</table>



<h3 id="add_metric"><code>add_metric</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L1416-L1536">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_metric(
    value, name=None, **kwargs
)
</code></pre>

Adds metric tensor to the layer.

This method can be used inside the `call()` method of a subclassed layer
or model.

```python
class MyMetricLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

  def call(self, inputs):
    self.add_metric(self.mean(inputs))
    self.add_metric(tf.reduce_sum(inputs), name='metric_2')
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any tensor passed to this Model must
be symbolic and be able to be traced back to the model's `Input`s. These
metrics become part of the model's topology and are tracked when you
save the model via `save()`.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(math_ops.reduce_sum(x), name='metric_1')
```

Note: Calling `add_metric()` with the result of a metric object on a
Functional Model, as shown in the example below, is not supported. This is
because we cannot trace the metric result tensor back to the model's inputs.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
Metric tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String metric name.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments for backward compatibility.
Accepted values:
`aggregation` - When the `value` tensor provided is not the result of
calling a `keras.Metric` instance, it will be aggregated by default
using a `keras.Metric.Mean`.
</td>
</tr>
</table>



<h3 id="add_weight"><code>add_weight</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L530-L703">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_weight(
    name=None,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    constraint=None,
    use_resource=None,
    synchronization=<a href="../../../tf/VariableSynchronization.md#AUTO"><code>tf.VariableSynchronization.AUTO</code></a>,
    aggregation=<a href="../../../tf/VariableAggregation.md#NONE"><code>tf.VariableAggregation.NONE</code></a>,
    **kwargs
)
</code></pre>

Adds a new variable to the layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
Variable name.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
Variable shape. Defaults to scalar if unspecified.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The type of the variable. Defaults to `self.dtype`.
</td>
</tr><tr>
<td>
`initializer`
</td>
<td>
Initializer instance (callable).
</td>
</tr><tr>
<td>
`regularizer`
</td>
<td>
Regularizer instance (callable).
</td>
</tr><tr>
<td>
`trainable`
</td>
<td>
Boolean, whether the variable should be part of the layer's
"trainable_variables" (e.g. variables, biases)
or "non_trainable_variables" (e.g. BatchNorm mean and variance).
Note that `trainable` cannot be `True` if `synchronization`
is set to `ON_READ`.
</td>
</tr><tr>
<td>
`constraint`
</td>
<td>
Constraint instance (callable).
</td>
</tr><tr>
<td>
`use_resource`
</td>
<td>
Whether to use `ResourceVariable`.
</td>
</tr><tr>
<td>
`synchronization`
</td>
<td>
Indicates when a distributed a variable will be
aggregated. Accepted values are constants defined in the class
<a href="../../../tf/VariableSynchronization.md"><code>tf.VariableSynchronization</code></a>. By default the synchronization is set to
`AUTO` and the current `DistributionStrategy` chooses
when to synchronize. If `synchronization` is set to `ON_READ`,
`trainable` must not be set to `True`.
</td>
</tr><tr>
<td>
`aggregation`
</td>
<td>
Indicates how a distributed variable will be aggregated.
Accepted values are constants defined in the class
<a href="../../../tf/VariableAggregation.md"><code>tf.VariableAggregation</code></a>.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments. Accepted values are `getter`,
`collections`, `experimental_autocast` and `caching_device`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The variable created.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
When giving unsupported dtype and no initializer or when
trainable has been set to True with synchronization set as `ON_READ`.
</td>
</tr>
</table>



<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L462-L483">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build(
    input_shape
)
</code></pre>

Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of `Layer` or `Model`
can override if they need a state-creation step in-between
layer instantiation and layer call. It is invoked automatically before
the first execution of `call()`.

This is typically used to create the weights of `Layer` subclasses
(at the discretion of the subclass implementer).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Instance of `TensorShape`, or list of instances of
`TensorShape` if the layer expects a list of inputs
(one instance per input).
</td>
</tr>
</table>



<h3 id="call"><code>call</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L485-L528">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call(
    inputs, *args, **kwargs
)
</code></pre>

This is where the layer's logic lives.

The `call()` method may not create state (except in its first invocation,
wrapping the creation of variables or other resources in <a href="../../../tf/init_scope.md"><code>tf.init_scope()</code></a>).
It is recommended to create state in `__init__()`, or the `build()` method
that is called automatically before `call()` executes the first time.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor, or dict/list/tuple of input tensors.
The first positional `inputs` argument is subject to special rules:
- `inputs` must be explicitly passed. A layer cannot have zero
  arguments, and `inputs` cannot be provided via the default value
  of a keyword argument.
- NumPy array or Python scalar values in `inputs` get cast as tensors.
- Keras mask metadata is only collected from `inputs`.
- Layers are built (`build(input_shape)` method)
  using shape info from `inputs` only.
- `input_spec` compatibility is only checked against `inputs`.
- Mixed precision input casting is only applied to `inputs`.
  If a layer has tensor arguments in `*args` or `**kwargs`, their
  casting behavior in mixed precision should be handled manually.
- The SavedModel input specification is generated using `inputs` only.
- Integration with various ecosystem packages like TFMOT, TFLite,
  TF.js, etc is only supported for `inputs` and not for tensors in
  positional and keyword arguments.
</td>
</tr><tr>
<td>
`*args`
</td>
<td>
Additional positional arguments. May contain tensors, although
this is not recommended, for the reasons above.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments. May contain tensors, although
this is not recommended, for the reasons above.
The following optional keyword arguments are reserved:
- `training`: Boolean scalar tensor of Python boolean indicating
  whether the `call` is meant for training or inference.
- `mask`: Boolean input mask. If the layer's `call()` method takes a
  `mask` argument, its default value will be set to the mask generated
  for `inputs` by the previous layer (if `input` did come from a layer
  that generated a corresponding mask, i.e. if it came from a Keras
  layer with masking support).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tensor or list/tuple of tensors.
</td>
</tr>

</table>



<h3 id="compute_mask"><code>compute_mask</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L871-L891">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_mask(
    inputs, mask=None
)
</code></pre>

Computes an output mask tensor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Tensor or list of tensors.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
Tensor or list of tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None or a tensor (or list of tensors,
one per output tensor of the layer).
</td>
</tr>

</table>



<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L785-L829">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_output_shape(
    input_shape
)
</code></pre>

Computes the output shape of the layer.

This method will cause the layer's state to be built, if that has not
happened before. This requires that the layer will later be used with
inputs that match the input shape provided here.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Shape tuple (tuple of integers)
or list of shape tuples (one per output tensor of the layer).
Shape tuples can include None for free dimensions,
instead of an integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An input shape tuple.
</td>
</tr>

</table>



<h3 id="compute_output_signature"><code>compute_output_signature</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L831-L869">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_output_signature(
    input_signature
)
</code></pre>

Compute the output tensor signature of the layer based on the inputs.

Unlike a TensorShape object, a TensorSpec object contains both shape
and dtype information for a tensor. This method allows layers to provide
output dtype information if it is different from the input dtype.
For any layer that doesn't implement this function,
the framework will fall back to use `compute_output_shape`, and will
assume that the output dtype matches the input dtype.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_signature`
</td>
<td>
Single TensorSpec or nested structure of TensorSpec
objects, describing a candidate input for the layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Single TensorSpec or nested structure of TensorSpec objects, describing
how the layer would transform the provided input.
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
If input_signature contains a non-TensorSpec object.
</td>
</tr>
</table>



<h3 id="count_params"><code>count_params</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L1948-L1968">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>count_params()
</code></pre>

Count the total number of scalars composing the weights.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An integer count.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the layer isn't yet built
(in which case its weights aren't yet defined).
</td>
</tr>
</table>



<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L767-L783">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Creates a layer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same layer from the config
dictionary. It does not handle layer connectivity
(handled by Network), nor weights (handled by `set_weights`).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary, typically the
output of get_config.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A layer instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L705-L765">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config of the layer.

A layer config is a Python dictionary (serializable)
containing the configuration of a layer.
The same layer can be reinstantiated later
(without its trained weights) from this configuration.

The config of a layer does not include connectivity
information, nor the layer class name. These are handled
by `Network` (one layer of abstraction above).

Note that `get_config()` does not guarantee to return a fresh copy of dict
every time it is called. The callers should make a copy of the returned dict
if they want to modify it.

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



<h3 id="get_weights"><code>get_weights</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L1656-L1698">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_weights()
</code></pre>

Returns the current weights of the layer, as NumPy arrays.

The weights of a layer represent the state of the layer. This function
returns both trainable and non-trainable weight values associated with this
layer as a list of NumPy arrays, which can in turn be used to load state
into similarly parameterized layers.

For example, a `Dense` layer returns a list of two values: the kernel matrix
and the bias vector. These can be used to set the weights of another
`Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Weights values as a list of NumPy arrays.
</td>
</tr>

</table>



<h3 id="set_weights"><code>set_weights</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L1570-L1654">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_weights(
    weights
)
</code></pre>

Sets the weights of the layer, from NumPy arrays.

The weights of a layer represent the state of the layer. This function
sets the weight values from numpy arrays. The weight values should be
passed in the order they are created by the layer. Note that the layer's
weights must be instantiated before calling this function, by calling
the layer.

For example, a `Dense` layer returns a list of two values: the kernel matrix
and the bias vector. These can be used to set the weights of another
`Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`weights`
</td>
<td>
a list of NumPy arrays. The number
of arrays and their shape must match
number of the dimensions of the weights
of the layer (i.e. it should match the
output of `get_weights`).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided weights list does not match the
layer's specifications.
</td>
</tr>
</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/engine/base_layer.py#L893-L1023">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Wraps `call`, applying pre- and post-processing steps.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
Positional arguments to be passed to `self.call`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to be passed to `self.call`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Output tensor(s).
</td>
</tr>

</table>



#### Note:

- The following optional keyword arguments are reserved for specific uses:
  * `training`: Boolean scalar tensor of Python boolean indicating
    whether the `call` is meant for training or inference.
  * `mask`: Boolean input mask.
- If the layer's `call` method takes a `mask` argument (as some Keras
  layers do), its default value will be set to the mask generated
  for `inputs` by the previous layer (if `input` did come from
  a layer that generated a corresponding mask, i.e. if it came from
  a Keras layer with masking support.
- If the layer is not built, the method will call `build`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the layer's `call` method returns None (an invalid value).
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
if `super().__init__()` was not called in the constructor.
</td>
</tr>
</table>





