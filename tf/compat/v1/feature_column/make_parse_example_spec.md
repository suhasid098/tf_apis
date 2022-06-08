description: Creates parsing spec dictionary from input feature_columns.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.feature_column.make_parse_example_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.feature_column.make_parse_example_spec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/feature_column.py">View source</a>



Creates parsing spec dictionary from input feature_columns.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.feature_column.make_parse_example_spec(
    feature_columns
)
</code></pre>



<!-- Placeholder for "Used in" -->

The returned dictionary can be used as arg 'features' in
<a href="../../../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a>.

#### Typical usage example:



```python
# Define features and transformations
feature_a = categorical_column_with_vocabulary_file(...)
feature_b = numeric_column(...)
feature_c_bucketized = bucketized_column(numeric_column("feature_c"), ...)
feature_a_x_feature_c = crossed_column(
    columns=["feature_a", feature_c_bucketized], ...)

feature_columns = set(
    [feature_b, feature_c_bucketized, feature_a_x_feature_c])
features = tf.io.parse_example(
    serialized=serialized_examples,
    features=make_parse_example_spec(feature_columns))
```

For the above example, make_parse_example_spec would return the dict:

```python
{
    "feature_a": parsing_ops.VarLenFeature(tf.string),
    "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
    "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
}
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
An iterable containing all feature columns. All items
should be instances of classes derived from `_FeatureColumn`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict mapping each feature key to a `FixedLenFeature` or `VarLenFeature`
value.
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
If any of the given `feature_columns` is not a `_FeatureColumn`
instance.
</td>
</tr>
</table>

