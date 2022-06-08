description: MultiHeadAttention layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.MultiHeadAttention" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.MultiHeadAttention

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/attention/multi_head_attention.py#L122-L515">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



MultiHeadAttention layer.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.MultiHeadAttention`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.MultiHeadAttention(
    num_heads,
    key_dim,
    value_dim=None,
    dropout=0.0,
    use_bias=True,
    output_shape=None,
    attention_axes=None,
    kernel_initializer=&#x27;glorot_uniform&#x27;,
    bias_initializer=&#x27;zeros&#x27;,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is an implementation of multi-headed attention as described in the paper
"Attention is all you Need" (Vaswani et al., 2017).
If `query`, `key,` `value` are the same, then
this is self-attention. Each timestep in `query` attends to the
corresponding sequence in `key`, and returns a fixed-width vector.

This layer first projects `query`, `key` and `value`. These are
(effectively) a list of tensors of length `num_attention_heads`, where the
corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
`(batch_size, <key/value dimensions>, key_dim)`,
`(batch_size, <key/value dimensions>, value_dim)`.

Then, the query and key tensors are dot-producted and scaled. These are
softmaxed to obtain attention probabilities. The value tensors are then
interpolated by these probabilities, then concatenated back to a single
tensor.

Finally, the result tensor with the last dimension as value_dim can take an
linear projection and return.

When using MultiHeadAttention inside a custom Layer, the custom Layer must
implement `build()` and call MultiHeadAttention's `_build_from_signature()`.
This enables weights to be restored correctly when the model is loaded.

when used in a custom Layer.

#### Examples:



Performs 1D cross-attention over two sequence inputs with an attention mask.
Returns the additional attention weights over heads.

```
>>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
>>> target = tf.keras.Input(shape=[8, 16])
>>> source = tf.keras.Input(shape=[4, 16])
>>> output_tensor, weights = layer(target, source,
...                                return_attention_scores=True)
>>> print(output_tensor.shape)
(None, 8, 16)
>>> print(weights.shape)
(None, 2, 8, 4)
```

Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

```
>>> layer = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
>>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
>>> output_tensor = layer(input_tensor, input_tensor)
>>> print(output_tensor.shape)
(None, 5, 3, 4, 16)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_heads`
</td>
<td>
Number of attention heads.
</td>
</tr><tr>
<td>
`key_dim`
</td>
<td>
Size of each attention head for query and key.
</td>
</tr><tr>
<td>
`value_dim`
</td>
<td>
Size of each attention head for value.
</td>
</tr><tr>
<td>
`dropout`
</td>
<td>
Dropout probability.
</td>
</tr><tr>
<td>
`use_bias`
</td>
<td>
Boolean, whether the dense layers use bias vectors/matrices.
</td>
</tr><tr>
<td>
`output_shape`
</td>
<td>
The expected shape of an output tensor, besides the batch and
sequence dims. If not specified, projects back to the key feature dim.
</td>
</tr><tr>
<td>
`attention_axes`
</td>
<td>
axes over which the attention is applied. `None` means
attention over all axes, but batch, heads, and features.
</td>
</tr><tr>
<td>
`kernel_initializer`
</td>
<td>
Initializer for dense layer kernels.
</td>
</tr><tr>
<td>
`bias_initializer`
</td>
<td>
Initializer for dense layer biases.
</td>
</tr><tr>
<td>
`kernel_regularizer`
</td>
<td>
Regularizer for dense layer kernels.
</td>
</tr><tr>
<td>
`bias_regularizer`
</td>
<td>
Regularizer for dense layer biases.
</td>
</tr><tr>
<td>
`activity_regularizer`
</td>
<td>
Regularizer for dense layer activity.
</td>
</tr><tr>
<td>
`kernel_constraint`
</td>
<td>
Constraint for dense layer kernels.
</td>
</tr><tr>
<td>
`bias_constraint`
</td>
<td>
Constraint for dense layer kernels.
</td>
</tr>
</table>



#### Call arguments:


* <b>`query`</b>: Query `Tensor` of shape `(B, T, dim)`.
* <b>`value`</b>: Value `Tensor` of shape `(B, S, dim)`.
* <b>`key`</b>: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
  `value` for both `key` and `value`, which is the most common case.
* <b>`attention_mask`</b>: a boolean mask of shape `(B, T, S)`, that prevents
  attention to certain positions. The boolean mask specifies which query
  elements can attend to which key elements, 1 indicates attention and 0
  indicates no attention. Broadcasting can happen for the missing batch
  dimensions and the head dimension.
* <b>`return_attention_scores`</b>: A boolean to indicate whether the output should
  be `(attention_output, attention_scores)` if `True`, or `attention_output`
  if `False`. Defaults to `False`.
* <b>`training`</b>: Python boolean indicating whether the layer should behave in
  training mode (adding dropout) or in inference mode (no dropout).
  Defaults to either using the training mode of the parent layer/model,
  or False (inference) if there is no parent layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`attention_output`
</td>
<td>
The result of the computation, of shape `(B, T, E)`,
where `T` is for target sequence shapes and `E` is the query input last
dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
are project to the shape specified by `output_shape`.
</td>
</tr><tr>
<td>
`attention_scores`
</td>
<td>
[Optional] multi-head attention coefficients over
attention axes.
</td>
</tr>
</table>



