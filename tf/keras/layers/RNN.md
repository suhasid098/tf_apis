description: Base class for recurrent layers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.RNN" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="reset_states"/>
</div>

# tf.keras.layers.RNN

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/rnn/base_rnn.py#L35-L862">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base class for recurrent layers.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.RNN`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.RNN(
    cell,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    time_major=False,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
for details about the usage of RNN API.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cell`
</td>
<td>
A RNN cell instance or a list of RNN cell instances.
A RNN cell is a class that has:
- A `call(input_at_t, states_at_t)` method, returning
  `(output_at_t, states_at_t_plus_1)`. The call method of the
  cell can also take the optional argument `constants`, see
  section "Note on passing external constants" below.
- A `state_size` attribute. This can be a single integer
  (single state) in which case it is the size of the recurrent
  state. This can also be a list/tuple of integers (one size per state).
  The `state_size` can also be TensorShape or tuple/list of
  TensorShape, to represent high dimension state.
- A `output_size` attribute. This can be a single integer or a
  TensorShape, which represent the shape of the output. For backward
  compatible reason, if this attribute is not available for the
  cell, the value will be inferred by the first element of the
  `state_size`.
- A `get_initial_state(inputs=None, batch_size=None, dtype=None)`
  method that creates a tensor meant to be fed to `call()` as the
  initial state, if the user didn't specify any initial state via other
  means. The returned initial state should have a shape of
  [batch_size, cell.state_size]. The cell might choose to create a
  tensor full of zeros, or full of other values based on the cell's
  implementation.
  `inputs` is the input tensor to the RNN layer, which should
  contain the batch size as its shape[0], and also dtype. Note that
  the shape[0] might be `None` during the graph construction. Either
  the `inputs` or the pair of `batch_size` and `dtype` are provided.
  `batch_size` is a scalar tensor that represents the batch size
  of the inputs. `dtype` is <a href="../../../tf/dtypes/DType.md"><code>tf.DType</code></a> that represents the dtype of
  the inputs.
  For backward compatibility, if this method is not implemented
  by the cell, the RNN layer will create a zero filled tensor with the
  size of [batch_size, cell.state_size].
In the case that `cell` is a list of RNN cell instances, the cells
will be stacked on top of each other in the RNN, resulting in an
efficient stacked RNN.
</td>
</tr><tr>
<td>
`return_sequences`
</td>
<td>
Boolean (default `False`). Whether to return the last
output in the output sequence, or the full sequence.
</td>
</tr><tr>
<td>
`return_state`
</td>
<td>
Boolean (default `False`). Whether to return the last state
in addition to the output.
</td>
</tr><tr>
<td>
`go_backwards`
</td>
<td>
Boolean (default `False`).
If True, process the input sequence backwards and return the
reversed sequence.
</td>
</tr><tr>
<td>
`stateful`
</td>
<td>
Boolean (default `False`). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.
</td>
</tr><tr>
<td>
`unroll`
</td>
<td>
Boolean (default `False`).
If True, the network will be unrolled, else a symbolic loop will be used.
Unrolling can speed-up a RNN, although it tends to be more
memory-intensive. Unrolling is only suitable for short sequences.
</td>
</tr><tr>
<td>
`time_major`
</td>
<td>
The shape format of the `inputs` and `outputs` tensors.
If True, the inputs and outputs will be in shape
`(timesteps, batch, ...)`, whereas in the False case, it will be
`(batch, timesteps, ...)`. Using `time_major = True` is a bit more
efficient because it avoids transposes at the beginning and end of the
RNN calculation. However, most TensorFlow data is batch-major, so by
default this function accepts input and emits output in batch-major
form.
</td>
</tr><tr>
<td>
`zero_output_for_mask`
</td>
<td>
Boolean (default `False`).
Whether the output should use zeros for the masked timesteps. Note that
this field is only used when `return_sequences` is True and mask is
provided. It can useful if you want to reuse the raw output sequence of
the RNN without interference from the masked timesteps, eg, merging
bidirectional RNNs.
</td>
</tr>
</table>



#### Call arguments:


* <b>`inputs`</b>: Input tensor.
* <b>`mask`</b>: Binary tensor of shape `[batch_size, timesteps]` indicating whether
  a given timestep should be masked. An individual `True` entry indicates
  that the corresponding timestep should be utilized, while a `False`
  entry indicates that the corresponding timestep should be ignored.
* <b>`training`</b>: Python boolean indicating whether the layer should behave in
  training mode or in inference mode. This argument is passed to the cell
  when calling it. This is for use with cells that use dropout.
* <b>`initial_state`</b>: List of initial state tensors to be passed to the first
  call of the cell.
* <b>`constants`</b>: List of constant tensors to be passed to the cell at each
  timestep.


#### Input shape:

N-D tensor with shape `[batch_size, timesteps, ...]` or
`[timesteps, batch_size, ...]` when time_major is True.



#### Output shape:

- If `return_state`: a list of tensors. The first tensor is
  the output. The remaining tensors are the last states,
  each with shape `[batch_size, state_size]`, where `state_size` could
  be a high dimension tensor shape.
- If `return_sequences`: N-D tensor with shape
  `[batch_size, timesteps, output_size]`, where `output_size` could
  be a high dimension tensor shape, or
  `[timesteps, batch_size, output_size]` when `time_major` is True.
- Else, N-D tensor with shape `[batch_size, output_size]`, where
  `output_size` could be a high dimension tensor shape.



#### Masking:

This layer supports masking for input data with a variable number
of timesteps. To introduce masks to your data,
use an [tf.keras.layers.Embedding] layer with the `mask_zero` parameter
set to `True`.


Note on using statefulness in RNNs:
  You can set RNN layers to be 'stateful', which means that the states
  computed for the samples in one batch will be reused as initial states
  for the samples in the next batch. This assumes a one-to-one mapping
  between samples in different successive batches.

  To enable statefulness:
    - Specify `stateful=True` in the layer constructor.
    - Specify a fixed batch size for your model, by passing
      If sequential model:
        `batch_input_shape=(...)` to the first layer in your model.
      Else for functional model with 1 or more Input layers:
        `batch_shape=(...)` to all the first layers in your model.
      This is the expected shape of your inputs
      *including the batch size*.
      It should be a tuple of integers, e.g. `(32, 10, 100)`.
    - Specify `shuffle=False` when calling `fit()`.

  To reset the states of your model, call `.reset_states()` on either
  a specific layer, or on your entire model.

Note on specifying the initial state of RNNs:
  You can specify the initial state of RNN layers symbolically by
  calling them with the keyword argument `initial_state`. The value of
  `initial_state` should be a tensor or list of tensors representing
  the initial state of the RNN layer.

  You can specify the initial state of RNN layers numerically by
  calling `reset_states` with the keyword argument `states`. The value of
  `states` should be a numpy array or list of numpy arrays representing
  the initial state of the RNN layer.

Note on passing external constants to RNNs:
  You can pass "external" constants to the cell using the `constants`
  keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
  requires that the `cell.call` method accepts the same keyword argument
  `constants`. Such constants can be used to condition the cell
  transformation on additional static inputs (not changing over time),
  a.k.a. an attention mechanism.

#### Examples:



```python
# First, let's define a RNN Cell, as a layer subclass.

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# Let's use this cell in a RNN layer:

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`states`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="reset_states"><code>reset_states</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/rnn/base_rnn.py#L752-L831">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_states(
    states=None
)
</code></pre>

Reset the recorded states for the stateful RNN layer.

Can only be used when RNN layer is constructed with `stateful` = `True`.
Args:
  states: Numpy arrays that contains the value for the initial state, which
    will be feed to cell at the first time step. When the value is None,
    zero filled numpy array will be created based on the cell state size.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`AttributeError`
</td>
<td>
When the RNN layer is not stateful.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
When the batch size of the RNN layer is unknown.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
When the input numpy array is not compatible with the RNN
layer state, either size wise or dtype wise.
</td>
</tr>
</table>





