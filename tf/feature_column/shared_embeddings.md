description: List of dense columns that convert from sparse, categorical input.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.feature_column.shared_embeddings" />
<meta itemprop="path" content="Stable" />
</div>

# tf.feature_column.shared_embeddings

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/feature_column_v2.py">View source</a>



List of dense columns that convert from sparse, categorical input.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.feature_column.shared_embeddings(
    categorical_columns,
    dimension,
    combiner=&#x27;mean&#x27;,
    initializer=None,
    shared_embedding_collection_name=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True,
    use_safe_embedding_lookup=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is similar to `embedding_column`, except that it produces a list of
embedding columns that share the same embedding weights.

Use this when your inputs are sparse and of the same type (e.g. watched and
impression video IDs that share the same vocabulary), and you want to convert
them to a dense representation (e.g., to feed to a DNN).

Inputs must be a list of categorical columns created by any of the
`categorical_column_*` function. They must all be of the same type and have
the same arguments except `key`. E.g. they can be
categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
all columns could also be weighted_categorical_column.

Here is an example embedding of two features for a DNNClassifier model:

```python
watched_video_id = categorical_column_with_vocabulary_file(
    'watched_video_id', video_vocabulary_file, video_vocabulary_size)
impression_video_id = categorical_column_with_vocabulary_file(
    'impression_video_id', video_vocabulary_file, video_vocabulary_size)
columns = shared_embedding_columns(
    [watched_video_id, impression_video_id], dimension=10)

estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

label_column = ...
def input_fn():
  features = tf.io.parse_example(
      ..., features=make_parse_example_spec(columns + [label_column]))
  labels = features.pop(label_column.name)
  return features, labels

estimator.train(input_fn=input_fn, steps=100)
```

Here is an example using `shared_embedding_columns` with model_fn:

```python
def model_fn(features, ...):
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  columns = shared_embedding_columns(
      [watched_video_id, impression_video_id], dimension=10)
  dense_tensor = input_layer(features, columns)
  # Form DNN layers, calculate loss, and return EstimatorSpec.
  ...
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`categorical_columns`
</td>
<td>
List of categorical columns created by a
`categorical_column_with_*` function. These columns produce the sparse IDs
that are inputs to the embedding lookup. All columns must be of the same
type and have the same arguments except `key`. E.g. they can be
categorical_column_with_vocabulary_file with the same vocabulary_file.
Some or all columns could also be weighted_categorical_column.
</td>
</tr><tr>
<td>
`dimension`
</td>
<td>
An integer specifying dimension of the embedding, must be > 0.
</td>
</tr><tr>
<td>
`combiner`
</td>
<td>
A string specifying how to reduce if there are multiple entries
in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
'mean' the default. 'sqrtn' often achieves good accuracy, in particular
with bag-of-words columns. Each of this can be thought as example level
normalizations on the column. For more information, see
`tf.embedding_lookup_sparse`.
</td>
</tr><tr>
<td>
`initializer`
</td>
<td>
A variable initializer function to be used in embedding
variable initialization. If not specified, defaults to
`truncated_normal_initializer` with mean `0.0` and standard
deviation `1/sqrt(dimension)`.
</td>
</tr><tr>
<td>
`shared_embedding_collection_name`
</td>
<td>
Optional collective name of these columns.
If not given, a reasonable name will be chosen based on the names of
`categorical_columns`.
</td>
</tr><tr>
<td>
`ckpt_to_load_from`
</td>
<td>
String representing checkpoint name/pattern from which to
restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
</td>
</tr><tr>
<td>
`tensor_name_in_ckpt`
</td>
<td>
Name of the `Tensor` in `ckpt_to_load_from` from
which to restore the column weights. Required if `ckpt_to_load_from` is
not `None`.
</td>
</tr><tr>
<td>
`max_norm`
</td>
<td>
If not `None`, each embedding is clipped if its l2-norm is
larger than this value, before combining.
</td>
</tr><tr>
<td>
`trainable`
</td>
<td>
Whether or not the embedding is trainable. Default is True.
</td>
</tr><tr>
<td>
`use_safe_embedding_lookup`
</td>
<td>
If true, uses safe_embedding_lookup_sparse
instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
there are no empty rows and all weights and ids are positive at the
expense of extra compute cost. This only applies to rank 2 (NxM) shaped
input tensors. Defaults to true, consider turning off if the above checks
are not needed. Note that having empty rows will not trigger any error
though the output result might be 0 or omitted.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of dense columns that converts from sparse input. The order of
results follows the ordering of `categorical_columns`.
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
if `dimension` not > 0.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
if any of the given `categorical_columns` is of different type
or has different arguments than the others.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
is specified.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
if `initializer` is specified and is not callable.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
if eager execution is enabled.
</td>
</tr>
</table>

