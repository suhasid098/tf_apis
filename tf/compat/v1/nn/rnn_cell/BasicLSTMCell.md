description: DEPRECATED: Please use <a href="../../../../../tf/compat/v1/nn/rnn_cell/LSTMCell.md"><code>tf.compat.v1.nn.rnn_cell.LSTMCell</code></a> instead.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.nn.rnn_cell.BasicLSTMCell" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="get_initial_state"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_updates_for"/>
<meta itemprop="property" content="zero_state"/>
</div>

# tf.compat.v1.nn.rnn_cell.BasicLSTMCell

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/rnn/legacy_cells.py#L616-L787">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DEPRECATED: Please use <a href="../../../../../tf/compat/v1/nn/rnn_cell/LSTMCell.md"><code>tf.compat.v1.nn.rnn_cell.LSTMCell</code></a> instead.

Inherits From: [`RNNCell`](../../../../../tf/compat/v1/nn/rnn_cell/RNNCell.md), [`Layer`](../../../../../tf/compat/v1/layers/Layer.md), [`Layer`](../../../../../tf/keras/layers/Layer.md), [`Module`](../../../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
    num_units,
    forget_bias=1.0,
    state_is_tuple=True,
    activation=None,
    reuse=None,
    name=None,
    dtype=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Basic LSTM recurrent network cell.

The implementation is based on

We add forget_bias (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.

It does not allow cell clipping, a projection layer, and does not
use peep-hole connections: it is the basic baseline.

For advanced models, please use the full <a href="../../../../../tf/compat/v1/nn/rnn_cell/LSTMCell.md"><code>tf.compat.v1.nn.rnn_cell.LSTMCell</code></a>
that follows.

Note that this cell is not optimized for performance. Please use
`tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
`tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
better performance on CPU.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_units`
</td>
<td>
int, The number of units in the LSTM cell.
</td>
</tr><tr>
<td>
`forget_bias`
</td>
<td>
float, The bias added to forget gates (see above). Must set
to `0.0` manually when restoring from CudnnLSTM-trained checkpoints.
</td>
</tr><tr>
<td>
`state_is_tuple`
</td>
<td>
If True, accepted and returned states are 2-tuples of the
`c_state` and `m_state`.  If False, they are concatenated along the
column axis.  The latter behavior will soon be deprecated.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
Activation function of the inner states.  Default: `tanh`. It
could also be string that is within Keras activation function names.
</td>
</tr><tr>
<td>
`reuse`
</td>
<td>
(optional) Python boolean describing whether to reuse variables in
an existing scope.  If not `True`, and the existing scope already has
the given variables, an error is raised.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String, the name of the layer. Layers with the same name will share
weights, but to avoid mistakes we require reuse=True in such cases.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Default dtype of the layer (default of `None` means use the type of
the first input). Required when `build` is called before `call`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Dict, keyword named properties for common layer attributes, like
`trainable` etc when constructing the cell from configs of get_config().
When restoring from CudnnLSTM-trained checkpoints, must use
`CudnnCompatibleLSTMCell` instead.
</td>
</tr>
</table>





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
`output_size`
</td>
<td>
Integer or TensorShape: size of outputs produced by this cell.
</td>
</tr><tr>
<td>
`scope_name`
</td>
<td>

</td>
</tr><tr>
<td>
`state_size`
</td>
<td>
size(s) of state(s) used by this cell.

It can be represented by an Integer, a TensorShape or a tuple of Integers
or TensorShapes.
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




<h3 id="get_initial_state"><code>get_initial_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/rnn/legacy_cells.py#L239-L266">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_initial_state(
    inputs=None, batch_size=None, dtype=None
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



<h3 id="zero_state"><code>zero_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/rnn/legacy_cells.py#L268-L297">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>zero_state(
    batch_size, dtype
)
</code></pre>

Return zero-filled state tensor(s).


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch_size`
</td>
<td>
int, float, or unit Tensor representing the batch size.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
the data type to use for the state.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
If `state_size` is an int or TensorShape, then the return value is a
`N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

If `state_size` is a nested list or tuple, then the return value is
a nested list or tuple (of the same structure) of `2-D` tensors with
the shapes `[batch_size, s]` for each s in `state_size`.
</td>
</tr>

</table>





