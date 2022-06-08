description: Class to keep track of the specification for TPU embeddings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.estimator.tpu.experimental.EmbeddingConfigSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tf.compat.v1.estimator.tpu.experimental.EmbeddingConfigSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/tpu/_tpu_estimator_embedding.py#L201-L386">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class to keep track of the specification for TPU embeddings.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.estimator.tpu.experimental.EmbeddingConfigSpec(
    feature_columns=None,
    optimization_parameters=None,
    clipping_limit=None,
    pipeline_execution_with_tensor_core=False,
    experimental_gradient_multiplier_fn=None,
    feature_to_config_dict=None,
    table_to_config_dict=None,
    partition_strategy=&#x27;div&#x27;,
    profile_data_directory=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

TPU Estimator manages its own TensorFlow graph and session, so it is not
compatible with TF2 behaviors. We recommend that you migrate to the newer
<a href="../../../../../../tf/distribute/TPUStrategy.md"><code>tf.distribute.TPUStrategy</code></a>. See the
[TPU guide](https://www.tensorflow.org/guide/tpu) for details.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Pass this class to `tf.estimator.tpu.TPUEstimator` via the
`embedding_config_spec` parameter. At minimum you need to specify
`feature_columns` and `optimization_parameters`. The feature columns passed
should be created with some combination of
`tf.tpu.experimental.embedding_column` and
`tf.tpu.experimental.shared_embedding_columns`.

TPU embeddings do not support arbitrary Tensorflow optimizers and the
main optimizer you use for your model will be ignored for the embedding table
variables. Instead TPU embeddigns support a fixed set of predefined optimizers
that you can select from and set the parameters of. These include adagrad,
adam and stochastic gradient descent. Each supported optimizer has a
`Parameters` class in the <a href="../../../../../../tf/tpu/experimental.md"><code>tf.tpu.experimental</code></a> namespace.

```
column_a = tf.feature_column.categorical_column_with_identity(...)
column_b = tf.feature_column.categorical_column_with_identity(...)
column_c = tf.feature_column.categorical_column_with_identity(...)
tpu_shared_columns = tf.tpu.experimental.shared_embedding_columns(
    [column_a, column_b], 10)
tpu_non_shared_column = tf.tpu.experimental.embedding_column(
    column_c, 10)
tpu_columns = [tpu_non_shared_column] + tpu_shared_columns
...
def model_fn(features):
  dense_features = tf.keras.layers.DenseFeature(tpu_columns)
  embedded_feature = dense_features(features)
  ...

estimator = tf.estimator.tpu.TPUEstimator(
    model_fn=model_fn,
    ...
    embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        column=tpu_columns,
        optimization_parameters=(
            tf.estimator.tpu.experimental.AdagradParameters(0.1))))
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
All embedding `FeatureColumn`s used by model.
</td>
</tr><tr>
<td>
`optimization_parameters`
</td>
<td>
An instance of `AdagradParameters`,
`AdamParameters` or `StochasticGradientDescentParameters`. This
optimizer will be applied to all embedding variables specified by
`feature_columns`.
</td>
</tr><tr>
<td>
`clipping_limit`
</td>
<td>
(Optional) Clipping limit (absolute value).
</td>
</tr><tr>
<td>
`pipeline_execution_with_tensor_core`
</td>
<td>
setting this to `True` makes training
faster, but trained model will be different if step N and step N+1
involve the same set of embedding IDs. Please see
`tpu_embedding_configuration.proto` for details.
</td>
</tr><tr>
<td>
`experimental_gradient_multiplier_fn`
</td>
<td>
(Optional) A Fn taking global step as
input returning the current multiplier for all embedding gradients.
</td>
</tr><tr>
<td>
`feature_to_config_dict`
</td>
<td>
A dictionary mapping feature names to instances of
the class `FeatureConfig`. Either features_columns or the pair of
`feature_to_config_dict` and `table_to_config_dict` must be specified.
</td>
</tr><tr>
<td>
`table_to_config_dict`
</td>
<td>
A dictionary mapping feature names to instances of
the class `TableConfig`. Either features_columns or the pair of
`feature_to_config_dict` and `table_to_config_dict` must be specified.
</td>
</tr><tr>
<td>
`partition_strategy`
</td>
<td>
A string, determining how tensors are sharded to the
tpu hosts. See <a href="../../../../../../tf/nn/safe_embedding_lookup_sparse.md"><code>tf.nn.safe_embedding_lookup_sparse</code></a> for more details.
Allowed value are `"div"` and `"mod"'. If `"mod"` is used, evaluation
and exporting the model to CPU will not work as expected.
</td>
</tr><tr>
<td>
`profile_data_directory`
</td>
<td>
Directory where embedding lookup statistics are
stored. These statistics summarize information about the inputs to the
embedding lookup operation, in particular, the average number of
embedding IDs per example and how well the embedding IDs are load
balanced across the system. The lookup statistics are used during TPU
initialization for embedding table partitioning. Collection of lookup
statistics is done at runtime by  profiling the embedding inputs, only a
small fraction of input samples are profiled to minimize host CPU
overhead. Once a suitable number of samples are profiled, the lookup
statistics are saved to table-specific files in the profile data
directory generally at the end of a TPU training loop. The filename
corresponding to each table is obtained by hashing table specific
parameters (e.g., table name and number of features) and global
configuration parameters (e.g., sharding strategy and task count). The
same profile data directory can be shared among several models to reuse
embedding lookup statistics.
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
If the feature_columns are not specified.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If the feature columns are not of ths correct type (one of
_SUPPORTED_FEATURE_COLUMNS, _TPU_EMBEDDING_COLUMN_CLASSES OR
_EMBEDDING_COLUMN_CLASSES).
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If `optimization_parameters` is not one of the required types.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`feature_columns`
</td>
<td>
A `namedtuple` alias for field number 0
</td>
</tr><tr>
<td>
`tensor_core_feature_columns`
</td>
<td>
A `namedtuple` alias for field number 1
</td>
</tr><tr>
<td>
`optimization_parameters`
</td>
<td>
A `namedtuple` alias for field number 2
</td>
</tr><tr>
<td>
`clipping_limit`
</td>
<td>
A `namedtuple` alias for field number 3
</td>
</tr><tr>
<td>
`pipeline_execution_with_tensor_core`
</td>
<td>
A `namedtuple` alias for field number 4
</td>
</tr><tr>
<td>
`experimental_gradient_multiplier_fn`
</td>
<td>
A `namedtuple` alias for field number 5
</td>
</tr><tr>
<td>
`feature_to_config_dict`
</td>
<td>
A `namedtuple` alias for field number 6
</td>
</tr><tr>
<td>
`table_to_config_dict`
</td>
<td>
A `namedtuple` alias for field number 7
</td>
</tr><tr>
<td>
`partition_strategy`
</td>
<td>
A `namedtuple` alias for field number 8
</td>
</tr><tr>
<td>
`profile_data_directory`
</td>
<td>
A `namedtuple` alias for field number 9
</td>
</tr>
</table>



