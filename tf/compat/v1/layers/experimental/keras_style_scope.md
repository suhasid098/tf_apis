description: Use Keras-style variable management.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.layers.experimental.keras_style_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.layers.experimental.keras_style_scope

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/legacy_tf_layers/base.py#L39-L107">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Use Keras-style variable management.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf_contextlib.contextmanager</code>
<code>tf.compat.v1.layers.experimental.keras_style_scope()
</code></pre>



<!-- Placeholder for "Used in" -->

All tf.layers and tf RNN cells created in this scope use Keras-style
variable management.  Creating such layers with a scope= argument is
disallowed, and reuse=True is disallowed.

The purpose of this scope is to allow users of existing layers to
slowly transition to a Keras layers API without breaking existing
functionality.

One example of this is when using TensorFlow's RNN classes with Keras
Models or Networks.  Because Keras models do not properly set variable
scopes, users of RNNs may either accidentally share scopes between two
different models, or get errors about variables that already exist.

#### Example:



```python
class RNNModel(tf.keras.Model):

  def __init__(self, name):
    super(RNNModel, self).__init__(name=name)
    self.rnn = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
      [tf.compat.v1.nn.rnn_cell.LSTMCell(64) for _ in range(2)])

  def call(self, input, state):
    return self.rnn(input, state)

model_1 = RNNModel("model_1")
model_2 = RNNModel("model_2")

# OK
output_1, next_state_1 = model_1(input, state)
# Raises an error about trying to create an already existing variable.
output_2, next_state_2 = model_2(input, state)
```

The solution is to wrap the model construction and execution in a keras-style
scope:

```python
with keras_style_scope():
  model_1 = RNNModel("model_1")
  model_2 = RNNModel("model_2")

  # model_1 and model_2 are guaranteed to create their own variables.
  output_1, next_state_1 = model_1(input, state)
  output_2, next_state_2 = model_2(input, state)

  assert len(model_1.weights) > 0
  assert len(model_2.weights) > 0
  assert(model_1.weights != model_2.weights)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
A keras layer style scope.
</td>
</tr>

</table>

