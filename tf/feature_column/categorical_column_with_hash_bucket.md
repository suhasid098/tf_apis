description: Represents sparse feature where ids are set by hashing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.feature_column.categorical_column_with_hash_bucket" />
<meta itemprop="path" content="Stable" />
</div>

# tf.feature_column.categorical_column_with_hash_bucket

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/feature_column_v2.py">View source</a>



Represents sparse feature where ids are set by hashing.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.feature_column.categorical_column_with_hash_bucket`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.feature_column.categorical_column_with_hash_bucket(
    key,
    hash_bucket_size,
    dtype=<a href="../../tf/dtypes.md#string"><code>tf.dtypes.string</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

Use this when your sparse features are in string or integer format, and you
want to distribute your inputs into a finite number of buckets by hashing.
output_id = Hash(input_feature_string) % bucket_size for string type input.
For int type input, the value is converted to its string representation first
and then hashed by the same formula.

For input dictionary `features`, `features[key]` is either `Tensor` or
`SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
and `''` for string, which will be dropped by this feature column.

#### Example:



```python
import tensorflow as tf
keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
10000)
columns = [keywords]
features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
'LSTM', 'Keras', 'RNN']])}
linear_prediction, _, _ = tf.compat.v1.feature_column.linear_model(features,
columns)

# or
import tensorflow as tf
keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
10000)
keywords_embedded = tf.feature_column.embedding_column(keywords, 16)
columns = [keywords_embedded]
features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
'LSTM', 'Keras', 'RNN']])}
input_layer = tf.keras.layers.DenseFeatures(columns)
dense_tensor = input_layer(features)
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
A unique string identifying the input feature. It is used as the
column name and the dictionary key for feature parsing configs, feature
`Tensor` objects, and feature columns.
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
A `HashedCategoricalColumn`.
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

