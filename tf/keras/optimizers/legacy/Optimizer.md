description: Base class for Keras optimizers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.legacy.Optimizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.optimizers.legacy.Optimizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/legacy/optimizer.py#L22-L24">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base class for Keras optimizers.

Inherits From: [`Optimizer`](../../../../tf/keras/optimizers/Optimizer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.optimizers.legacy.Optimizer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.legacy.Optimizer(
    name, gradient_aggregator=None, gradient_transformers=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

You should not use this class directly, but instead instantiate one of its
subclasses such as <a href="../../../../tf/keras/optimizers/SGD.md"><code>tf.keras.optimizers.SGD</code></a>, <a href="../../../../tf/keras/optimizers/Adam.md"><code>tf.keras.optimizers.Adam</code></a>, etc.

### Usage

```python
# Create an optimizer with the desired parameters.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# `loss` is a callable that takes no argument and returns the value
# to minimize.
loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
# In graph mode, returns op that minimizes the loss by updating the listed
# variables.
opt_op = opt.minimize(loss, var_list=[var1, var2])
opt_op.run()
# In eager mode, simply call minimize to update the list of variables.
opt.minimize(loss, var_list=[var1, var2])
```

### Usage in custom training loops

In Keras models, sometimes variables are created when the model is first
called, instead of construction time. Examples include 1) sequential models
without input shape pre-defined, or 2) subclassed models. Pass var_list as
callable in these cases.

#### Example:



```python
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(num_hidden, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
loss_fn = lambda: tf.keras.losses.mse(model(input), output)
var_list_fn = lambda: model.trainable_weights
for input, output in data:
  opt.minimize(loss_fn, var_list_fn)
```

### Processing gradients before applying them

Calling `minimize()` takes care of both computing the gradients and
applying them to the variables.  If you want to process the gradients
before applying them you can instead use the optimizer in three steps:

1.  Compute the gradients with <a href="../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a>.
2.  Process the gradients as you wish.
3.  Apply the processed gradients with `apply_gradients()`.

#### Example:



```python
# Create an optimizer.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)

# Compute the gradients for a list of variables.
with tf.GradientTape() as tape:
  loss = <call_loss_function>
vars = <list_of_variables>
grads = tape.gradient(loss, vars)

# Process the gradients, for example cap them, etc.
# capped_grads = [MyCapper(g) for g in grads]
processed_grads = [process_gradient(g) for g in grads]

# Ask the optimizer to apply the processed gradients.
opt.apply_gradients(zip(processed_grads, var_list))
```

### Use with <a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>

This optimizer class is <a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> aware, which means it
automatically sums gradients across all replicas. To average gradients,
you divide your loss by the global batch size, which is done
automatically if you use <a href="../../../../tf/keras.md"><code>tf.keras</code></a> built-in training or evaluation loops.
See the `reduction` argument of your loss which should be set to
<a href="../../../../tf/keras/losses/Reduction.md#SUM_OVER_BATCH_SIZE"><code>tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE</code></a> for averaging or
<a href="../../../../tf/keras/losses/Reduction.md#SUM"><code>tf.keras.losses.Reduction.SUM</code></a> for not.

To aggregate gradients yourself, call `apply_gradients` with
`experimental_aggregate_gradients` set to False. This is useful if you need to
process aggregated gradients.

If you are not using these and you want to average gradients, you should use
<a href="../../../../tf/math/reduce_sum.md"><code>tf.math.reduce_sum</code></a> to add up your per-example losses and then divide by the
global batch size. Note that when using <a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>, the first
component of a tensor's shape is the *replica-local* batch size, which is off
by a factor equal to the number of replicas being used to compute a single
step. As a result, using <a href="../../../../tf/math/reduce_mean.md"><code>tf.math.reduce_mean</code></a> will give the wrong answer,
resulting in gradients that can be many times too big.

### Variable Constraints

All Keras optimizers respect variable constraints. If constraint function is
passed to any variable, the constraint will be applied to the variable after
the gradient has been applied to the variable.
Important: If gradient is sparse tensor, variable constraint is not supported.

### Thread Compatibility

The entire optimizer is currently thread compatible, not thread-safe. The user
needs to perform synchronization if necessary.

### Slots

Many optimizer subclasses, such as `Adam` and `Adagrad` allocate and manage
additional variables associated with the variables to train.  These are called
<i>Slots</i>.  Slots have names and you can ask the optimizer for the names of
the slots that it uses.  Once you have a slot name you can ask the optimizer
for the variable it created to hold the slot value.

This can be useful if you want to log debug a training algorithm, report stats
about the slots, etc.

### Hyperparameters

These are arguments passed to the optimizer subclass constructor
(the `__init__` method), and then passed to `self._set_hyper()`.
They can be either regular Python values (like 1.0), tensors, or
callables. If they are callable, the callable will be called during
`apply_gradients()` to get the value for the hyper parameter.

Hyperparameters can be overwritten through user code:

#### Example:



```python
# Create an optimizer with the desired parameters.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# `loss` is a callable that takes no argument and returns the value
# to minimize.
loss = lambda: 3 * var1 + 2 * var2
# In eager mode, simply call minimize to update the list of variables.
opt.minimize(loss, var_list=[var1, var2])
# update learning rate
opt.learning_rate = 0.05
opt.minimize(loss, var_list=[var1, var2])
```

### Callable learning rate

Optimizer accepts a callable learning rate in two ways. The first way is
through built-in or customized
<a href="../../../../tf/keras/optimizers/schedules/LearningRateSchedule.md"><code>tf.keras.optimizers.schedules.LearningRateSchedule</code></a>. The schedule will be
called on each iteration with `schedule(iteration)`, a <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a>
owned by the optimizer.

#### Example:



```
>>> var = tf.Variable(np.random.random(size=(1,)))
>>> learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
... initial_learning_rate=.01, decay_steps=20, decay_rate=.1)
>>> opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
>>> loss = lambda: 3 * var
>>> opt.minimize(loss, var_list=[var])
<tf.Variable...
```

The second way is through a callable function that
does not accept any arguments.

#### Example:



```
>>> var = tf.Variable(np.random.random(size=(1,)))
>>> def lr_callable():
...   return .1
>>> opt = tf.keras.optimizers.SGD(learning_rate=lr_callable)
>>> loss = lambda: 3 * var
>>> opt.minimize(loss, var_list=[var])
<tf.Variable...
```

### Creating a custom optimizer

If you intend to create your own optimization algorithm, simply inherit from
this class and override the following methods:

  - `_resource_apply_dense` (update variable given gradient tensor is a dense
    <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a>)
  - `_resource_apply_sparse` (update variable given gradient tensor is a
    sparse <a href="../../../../tf/IndexedSlices.md"><code>tf.IndexedSlices</code></a>. The most common way for this to happen
    is if you are taking the gradient through a <a href="../../../../tf/gather.md"><code>tf.gather</code></a>.)
  - `_create_slots`
    (if your optimizer algorithm requires additional variables)
  - `get_config`
    (serialization of the optimizer, include all hyper parameters)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
String. The name to use for momentum accumulator weights created
by the optimizer.
</td>
</tr><tr>
<td>
`gradient_aggregator`
</td>
<td>
The function to use to aggregate gradients across
devices (when using <a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>). If `None`, defaults to
summing the gradients across devices. The function should accept and
return a list of `(gradient, variable)` tuples.
</td>
</tr><tr>
<td>
`gradient_transformers`
</td>
<td>
Optional. List of functions to use to transform
gradients before applying updates to Variables. The functions are
applied after `gradient_aggregator`. The functions should accept and
return a list of `(gradient, variable)` tuples.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments. Allowed arguments are `clipvalue`,
`clipnorm`, `global_clipnorm`.
If `clipvalue` (float) is set, the gradient of each weight
is clipped to be no higher than this value.
If `clipnorm` (float) is set, the gradient of each weight
is individually clipped so that its norm is no higher than this value.
If `global_clipnorm` (float) is set the gradient of all weights is
clipped so that their global norm is no higher than this value.
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
in case of any invalid argument.
</td>
</tr>
</table>



