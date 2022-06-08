description: A layer for sequence input.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.experimental.SequenceFeatures" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.experimental.SequenceFeatures

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/feature_column/sequence_feature_column.py#L32-L167">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A layer for sequence input.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.experimental.SequenceFeatures`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.experimental.SequenceFeatures(
    feature_columns, trainable=True, name=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

All `feature_columns` must be sequence dense columns with the same
`sequence_length`. The output of this method can be fed into sequence
networks, such as RNN.

The output of this method is a 3D `Tensor` of shape `[batch_size, T, D]`.
`T` is the maximum sequence length for this batch, which could differ from
batch to batch.

If multiple `feature_columns` are given with `Di` `num_elements` each, their
outputs are concatenated. So, the final `Tensor` has shape
`[batch_size, T, D0 + D1 + ... + Dn]`.

#### Example:



```python

import tensorflow as tf

# Behavior of some cells or feature columns may depend on whether we are in
# training or inference mode, e.g. applying dropout.
training = True
rating = tf.feature_column.sequence_numeric_column('rating')
watches = tf.feature_column.sequence_categorical_column_with_identity(
    'watches', num_buckets=1000)
watches_embedding = tf.feature_column.embedding_column(watches,
                                            dimension=10)
columns = [rating, watches_embedding]

features = {
 'rating': tf.sparse.from_dense([[1.0,1.1, 0, 0, 0],
                                             [2.0,2.1,2.2, 2.3, 2.5]]),
 'watches': tf.sparse.from_dense([[2, 85, 0, 0, 0],[33,78, 2, 73, 1]])
}

sequence_input_layer = tf.keras.experimental.SequenceFeatures(columns)
sequence_input, sequence_length = sequence_input_layer(
   features, training=training)
sequence_length_mask = tf.sequence_mask(sequence_length)
hidden_size = 32
rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
rnn_layer = tf.keras.layers.RNN(rnn_cell)
outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`feature_columns`
</td>
<td>
An iterable of dense sequence columns. Valid columns are
- `embedding_column` that wraps a `sequence_categorical_column_with_*`
- `sequence_numeric_column`.
</td>
</tr><tr>
<td>
`trainable`
</td>
<td>
Boolean, whether the layer's variables will be updated via
gradient descent during training.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name to give to the SequenceFeatures.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to construct a layer.
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
If any of the `feature_columns` is not a
`SequenceDenseColumn`.
</td>
</tr>
</table>



