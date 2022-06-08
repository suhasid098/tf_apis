description: Optimizer that implements the gradient descent algorithm.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.train.GradientDescentOptimizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="compute_gradients"/>
<meta itemprop="property" content="get_name"/>
<meta itemprop="property" content="get_slot"/>
<meta itemprop="property" content="get_slot_names"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="GATE_GRAPH"/>
<meta itemprop="property" content="GATE_NONE"/>
<meta itemprop="property" content="GATE_OP"/>
</div>

# tf.compat.v1.train.GradientDescentOptimizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/gradient_descent.py">View source</a>



Optimizer that implements the gradient descent algorithm.

Inherits From: [`Optimizer`](../../../../tf/compat/v1/train/Optimizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate, use_locking=False, name=&#x27;GradientDescent&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->
  

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate`
</td>
<td>
A Tensor or a floating point value.  The learning
rate to use.
</td>
</tr><tr>
<td>
`use_locking`
</td>
<td>
If True use locks for update operations.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name prefix for the operations created when applying
gradients. Defaults to "GradientDescent".
</td>
</tr>
</table>



## Methods

<h3 id="apply_gradients"><code>apply_gradients</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_gradients(
    grads_and_vars, global_step=None, name=None
)
</code></pre>

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

@compatibility(TF2)
#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name    | Note                       |
| :-------------------- | :-------------- | :------------------------- |
| `grads_and_vars`      | `grads_and_vars`| -                          |
| `global_step`         | Not supported.  | Use `optimizer.iterations` |
| `name`                | `name. `        | -                          |

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`grads_and_vars`
</td>
<td>
List of (gradient, variable) pairs as returned by
`compute_gradients()`.
</td>
</tr><tr>
<td>
`global_step`
</td>
<td>
Optional `Variable` to increment by one after the
variables have been updated.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the returned operation.  Default to the
name passed to the `Optimizer` constructor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An `Operation` that applies the specified gradients. If `global_step`
was not None, that operation also increments `global_step`.
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
`ValueError`
</td>
<td>
If none of the variables have gradients.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If you should use `_distributed_apply()` instead.
</td>
</tr>
</table>



<h3 id="compute_gradients"><code>compute_gradients</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_gradients(
    loss,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    grad_loss=None
)
</code></pre>

Compute gradients of `loss` for the variables in `var_list`.


 <section><devsite-expandable expanded>
 <h4 class="showalways">Migrate to TF2</h4>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../../tf/keras/optimizers/Optimizer.md"><code>tf.keras.optimizers.Optimizer</code></a> in TF2 does not provide a
`compute_gradients` method, and you should use a <a href="../../../../tf/GradientTape.md"><code>tf.GradientTape</code></a> to
obtain the gradients:

```python
@tf.function
def train step(inputs):
  batch_data, labels = inputs
  with tf.GradientTape() as tape:
    predictions = model(batch_data, training=True)
    loss = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

Args:
  loss: A Tensor containing the value to minimize or a callable taking
    no arguments which returns the value to minimize. When eager execution
    is enabled it must be a callable.
  var_list: Optional list or tuple of <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a> to update to minimize
    `loss`.  Defaults to the list of variables collected in the graph
    under the key `GraphKeys.TRAINABLE_VARIABLES`.
  gate_gradients: How to gate the computation of gradients.  Can be
    `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
  aggregation_method: Specifies the method used to combine gradient terms.
    Valid values are defined in the class `AggregationMethod`.
  colocate_gradients_with_ops: If True, try colocating gradients with
    the corresponding op.
  grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

Returns:
  A list of (gradient, variable) pairs. Variable is always present, but
  gradient can be `None`.

Raises:
  TypeError: If `var_list` contains anything else than `Variable` objects.
  ValueError: If some arguments are invalid.
  RuntimeError: If called with eager execution enabled and `loss` is
    not callable.

@compatibility(eager)
When eager execution is enabled, `gate_gradients`, `aggregation_method`,
and `colocate_gradients_with_ops` are ignored.


 </aside></devsite-expandable></section>

<h4>Description</h4>


This is the first part of `minimize()`.  It returns a list
of (gradient, variable) pairs where "gradient" is the gradient
for "variable".  Note that "gradient" can be a `Tensor`, an
`IndexedSlices`, or `None` if there is no gradient for the
given variable.



<h3 id="get_name"><code>get_name</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_name()
</code></pre>




<h3 id="get_slot"><code>get_slot</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_slot(
    var, name
)
</code></pre>

Return a slot named `name` created for `var` by the Optimizer.

Some `Optimizer` subclasses use additional variables.  For example
`Momentum` and `Adagrad` use variables to accumulate updates.  This method
gives access to these `Variable` objects if for some reason you need them.

Use `get_slot_names()` to get the list of slot names created by the
`Optimizer`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var`
</td>
<td>
A variable passed to `minimize()` or `apply_gradients()`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The `Variable` for the slot if it was created, `None` otherwise.
</td>
</tr>

</table>



<h3 id="get_slot_names"><code>get_slot_names</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_slot_names()
</code></pre>

Return a list of the names of slots created by the `Optimizer`.

See `get_slot()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of strings.
</td>
</tr>

</table>



<h3 id="minimize"><code>minimize</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name=None,
    grad_loss=None
)
</code></pre>

Add operations to minimize `loss` by updating `var_list`.

This method simply combines calls `compute_gradients()` and
`apply_gradients()`. If you want to process the gradient before applying
them call `compute_gradients()` and `apply_gradients()` explicitly instead
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
A `Tensor` containing the value to minimize.
</td>
</tr><tr>
<td>
`global_step`
</td>
<td>
Optional `Variable` to increment by one after the
variables have been updated.
</td>
</tr><tr>
<td>
`var_list`
</td>
<td>
Optional list or tuple of `Variable` objects to update to
minimize `loss`.  Defaults to the list of variables collected in
the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
</td>
</tr><tr>
<td>
`gate_gradients`
</td>
<td>
How to gate the computation of gradients.  Can be
`GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
</td>
</tr><tr>
<td>
`aggregation_method`
</td>
<td>
Specifies the method used to combine gradient terms.
Valid values are defined in the class `AggregationMethod`.
</td>
</tr><tr>
<td>
`colocate_gradients_with_ops`
</td>
<td>
If True, try colocating gradients with
the corresponding op.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the returned operation.
</td>
</tr><tr>
<td>
`grad_loss`
</td>
<td>
Optional. A `Tensor` holding the gradient computed for `loss`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An Operation that updates the variables in `var_list`.  If `global_step`
was not `None`, that operation also increments `global_step`.
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
If some of the variables are not `Variable` objects.
</td>
</tr>
</table>




 <section><devsite-expandable expanded>
 <h4 class="showalways">eager compatibility</h4>

When eager execution is enabled, `loss` should be a Python function that
takes no arguments and computes the value to be minimized. Minimization (and
gradient computation) is done with respect to the elements of `var_list` if
not None, else with respect to any trainable variables created during the
execution of the `loss` function. `gate_gradients`, `aggregation_method`,
`colocate_gradients_with_ops` and `grad_loss` are ignored when eager
execution is enabled.


 </devsite-expandable></section>



<h3 id="variables"><code>variables</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>variables()
</code></pre>

A list of variables which encode the current state of `Optimizer`.

Includes slot variables and additional global variables created by the
optimizer in the current default graph.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of variables.
</td>
</tr>

</table>







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
GATE_GRAPH<a id="GATE_GRAPH"></a>
</td>
<td>
`2`
</td>
</tr><tr>
<td>
GATE_NONE<a id="GATE_NONE"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
GATE_OP<a id="GATE_OP"></a>
</td>
<td>
`1`
</td>
</tr>
</table>

