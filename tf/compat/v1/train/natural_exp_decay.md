description: Applies natural exponential decay to the initial learning rate.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.train.natural_exp_decay" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.train.natural_exp_decay

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/keras/optimizer_v2/legacy_learning_rate_decay.py">View source</a>



Applies natural exponential decay to the initial learning rate.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.train.natural_exp_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

When training a model, it is often recommended to lower the learning rate as
the training progresses.  This function applies an exponential decay function
to a provided initial learning rate.  It requires an `global_step` value to
compute the decayed learning rate.  You can just pass a TensorFlow variable
that you increment at each training step.

The function returns the decayed learning rate.  It is computed as:

```python
decayed_learning_rate = learning_rate * exp(-decay_rate * global_step /
decay_step)
```

or, if `staircase` is `True`, as:

```python
decayed_learning_rate = learning_rate * exp(-decay_rate * floor(global_step /
decay_step))
```

Example: decay exponentially with a base of 0.96:

```python
...
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1
decay_steps = 5
k = 0.5
learning_rate = tf.compat.v1.train.natural_exp_decay(learning_rate,
global_step,
                                           decay_steps, k)

# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)
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
A scalar `float32` or `float64` `Tensor` or a Python number.
The initial learning rate.
</td>
</tr><tr>
<td>
`global_step`
</td>
<td>
A Python number. Global step to use for the decay computation.
Must not be negative.
</td>
</tr><tr>
<td>
`decay_steps`
</td>
<td>
How often to apply decay.
</td>
</tr><tr>
<td>
`decay_rate`
</td>
<td>
A Python number.  The decay rate.
</td>
</tr><tr>
<td>
`staircase`
</td>
<td>
Whether to apply decay in a discrete staircase, as opposed to
continuous, fashion.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String.  Optional name of the operation.  Defaults to
'ExponentialTimeDecay'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar `Tensor` of the same type as `learning_rate`.  The decayed
learning rate.
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
if `global_step` is not supplied.
</td>
</tr>
</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

When eager execution is enabled, this function returns a function which in
turn returns the decayed learning rate Tensor. This can be useful for changing
the learning rate value across different invocations of optimizer functions.


 </devsite-expandable></section>

