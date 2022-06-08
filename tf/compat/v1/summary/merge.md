description: Merges summaries.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.summary.merge" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.summary.merge

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/summary/summary.py">View source</a>



Merges summaries.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.summary.merge(
    inputs, collections=None, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is not compatible with eager execution or <a href="../../../../tf/function.md"><code>tf.function</code></a>. To migrate
to TF2, this API can be omitted entirely, because in TF2 individual summary
ops, like <a href="../../../../tf/summary/scalar.md"><code>tf.summary.scalar()</code></a>, write directly to the default summary writer
if one is active. Thus, it's not necessary to merge summaries or to manually
add the resulting merged summary output to the writer. See the usage example
shown below.

For a comprehensive <a href="../../../../tf/summary.md"><code>tf.summary</code></a> migration guide, please follow
[Migrating tf.summary usage to
TF 2.0](https://www.tensorflow.org/tensorboard/migrate#in_tf_1x).

#### TF1 & TF2 Usage Example

TF1:

```python
dist = tf.compat.v1.placeholder(tf.float32, [100])
tf.compat.v1.summary.histogram(name="distribution", values=dist)
writer = tf.compat.v1.summary.FileWriter("/tmp/tf1_summary_example")
summaries = tf.compat.v1.summary.merge_all()

sess = tf.compat.v1.Session()
for step in range(100):
  mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
  summ = sess.run(summaries, feed_dict={dist: mean_moving_normal})
  writer.add_summary(summ, global_step=step)
```

TF2:

```python
writer = tf.summary.create_file_writer("/tmp/tf2_summary_example")
for step in range(100):
  mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
  with writer.as_default(step=step):
    tf.summary.histogram(name='distribution', data=mean_moving_normal)
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
A list of `string` `Tensor` objects containing serialized `Summary`
protocol buffers.
</td>
</tr><tr>
<td>
`collections`
</td>
<td>
Optional list of graph collections keys. The new summary op is
added to these collections. Defaults to `[]`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar `Tensor` of type `string`. The serialized `Summary` protocol
buffer resulting from the merging.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If called with eager mode enabled.
</td>
</tr>
</table>


