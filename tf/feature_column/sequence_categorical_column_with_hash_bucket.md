description: A sequence of categorical terms where ids are set by hashing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.feature_column.sequence_categorical_column_with_hash_bucket" />
<meta itemprop="path" content="Stable" />
</div>

# tf.feature_column.sequence_categorical_column_with_hash_bucket

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/sequence_feature_column.py">View source</a>



A sequence of categorical terms where ids are set by hashing.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.feature_column.sequence_categorical_column_with_hash_bucket`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.feature_column.sequence_categorical_column_with_hash_bucket(
    key,
    hash_bucket_size,
    dtype=<a href="../../tf/dtypes.md#string"><code>tf.dtypes.string</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

Pass this to `embedding_column` or `indicator_column` to convert sequence
categorical data into dense representation for input to sequence NN, such as
RNN.

#### Example:



```python
tokens = sequence_categorical_column_with_hash_bucket(
    'tokens', hash_bucket_size=1000)
tokens_embedding = embedding_column(tokens, dimension=10)
columns = [tokens_embedding]

features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
sequence_feature_layer = SequenceFeatures(columns)
sequence_input, sequence_length = sequence_feature_layer(features)
sequence_length_mask = tf.sequence_mask(sequence_length)

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
`key`
</td>
<td>
A unique string identifying the input feature.
</td>
</tr><tr>
<td>
`hash_bucket_size`
</td>
<td>
An int > 1. The number of buckets.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The type of features. Only string and integer types are supported.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `SequenceCategoricalColumn`.
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
`hash_bucket_size` is not greater than 1.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`dtype` is neither string nor integer.
</td>
</tr>
</table>

