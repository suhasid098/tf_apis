description: Use this method to track nested keras models in a shim-decorated method.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.keras.utils.get_or_create_layer" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.keras.utils.get_or_create_layer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/legacy_tf_layers/variable_scope_shim.py#L953-L1011">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Use this method to track nested keras models in a shim-decorated method.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.keras.utils.get_or_create_layer(
    name, create_layer_method
)
</code></pre>



<!-- Placeholder for "Used in" -->

This method can be used within a `tf.keras.Layer`'s methods decorated by
the`track_tf1_style_variables` shim, to additionally track inner keras Model
objects created within the same method. The inner model's variables and losses
will be accessible via the outer model's `variables` and `losses` attributes.

This enables tracking of inner keras models using TF2 behaviors, with minimal
changes to existing TF1-style code.

#### Example:



```python
class NestedLayer(tf.keras.layers.Layer):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units

  def build_model(self):
    inp = tf.keras.Input(shape=(5, 5))
    dense_layer = tf.keras.layers.Dense(
        10, name="dense", kernel_regularizer="l2",
        kernel_initializer=tf.compat.v1.ones_initializer())
    model = tf.keras.Model(inputs=inp, outputs=dense_layer(inp))
    return model

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs):
    model = tf.compat.v1.keras.utils.get_or_create_layer(
        "dense_model", self.build_model)
    return model(inputs)
```
The inner model creation should be confined to its own zero-arg function,
which should be passed into this method. In TF1, this method will immediately
create and return the desired model, without any tracking.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
A name to give the nested layer to track.
</td>
</tr><tr>
<td>
`create_layer_method`
</td>
<td>
a Callable that takes no args and returns the nested
layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The created layer.
</td>
</tr>

</table>

