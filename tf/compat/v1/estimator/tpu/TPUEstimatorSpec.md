description: Ops and objects returned from a model_fn and passed to TPUEstimator.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.estimator.tpu.TPUEstimatorSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="as_estimator_spec"/>
</div>

# tf.compat.v1.estimator.tpu.TPUEstimatorSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py#L280-L395">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
    mode,
    predictions=None,
    loss=None,
    train_op=None,
    eval_metrics=None,
    export_outputs=None,
    scaffold_fn=None,
    host_call=None,
    training_hooks=None,
    evaluation_hooks=None,
    prediction_hooks=None
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
<a href="../../../../../tf/distribute/TPUStrategy.md"><code>tf.distribute.TPUStrategy</code></a>. See the
[TPU guide](https://www.tensorflow.org/guide/tpu) for details.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

See `EstimatorSpec` for `mode`, `predictions`, `loss`, `train_op`, and
`export_outputs`.

For evaluation, `eval_metrics `is a tuple of `metric_fn` and `tensors`, where
`metric_fn` runs on CPU to generate metrics and `tensors` represents the
`Tensor`s transferred from TPU system to CPU host and passed to `metric_fn`.
To be precise, TPU evaluation expects a slightly different signature from the
<a href="../../../../../tf/estimator/Estimator.md"><code>tf.estimator.Estimator</code></a>. While `EstimatorSpec.eval_metric_ops` expects a
dict, `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`.
The `tensors` could be a list of `Tensor`s or dict of names to `Tensor`s. The
`tensors` usually specify the model logits, which are transferred back from
TPU system to CPU host. All tensors must have be batch-major, i.e., the batch
size is the first dimension. Once all tensors are available at CPU host from
all shards, they are concatenated (on CPU) and passed as positional arguments
to the `metric_fn` if `tensors` is list or keyword arguments if `tensors` is
a dict. `metric_fn` takes the `tensors` and returns a dict from metric string
name to the result of calling a metric function, namely a `(metric_tensor,
update_op)` tuple. See `TPUEstimator` for MNIST example how to specify the
`eval_metrics`.

`scaffold_fn` is a function running on CPU to generate the `Scaffold`. This
function should not capture any Tensors in `model_fn`.

`host_call` is a tuple of a `function` and a list or dictionary of `tensors`
to pass to that function and returns a list of Tensors. `host_call` currently
works for train() and evaluate(). The Tensors returned by the function is
executed on the CPU on every step, so there is communication overhead when
sending tensors from TPU to CPU. To reduce the overhead, try reducing the
size of the tensors. The `tensors` are concatenated along their major (batch)
dimension, and so must be >= rank 1. The `host_call` is useful for writing
summaries with `tf.contrib.summary.create_file_writer`.





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`mode`
</td>
<td>
A `namedtuple` alias for field number 0
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
A `namedtuple` alias for field number 1
</td>
</tr><tr>
<td>
`loss`
</td>
<td>
A `namedtuple` alias for field number 2
</td>
</tr><tr>
<td>
`train_op`
</td>
<td>
A `namedtuple` alias for field number 3
</td>
</tr><tr>
<td>
`eval_metrics`
</td>
<td>
A `namedtuple` alias for field number 4
</td>
</tr><tr>
<td>
`export_outputs`
</td>
<td>
A `namedtuple` alias for field number 5
</td>
</tr><tr>
<td>
`scaffold_fn`
</td>
<td>
A `namedtuple` alias for field number 6
</td>
</tr><tr>
<td>
`host_call`
</td>
<td>
A `namedtuple` alias for field number 7
</td>
</tr><tr>
<td>
`training_hooks`
</td>
<td>
A `namedtuple` alias for field number 8
</td>
</tr><tr>
<td>
`evaluation_hooks`
</td>
<td>
A `namedtuple` alias for field number 9
</td>
</tr><tr>
<td>
`prediction_hooks`
</td>
<td>
A `namedtuple` alias for field number 10
</td>
</tr>
</table>



## Methods

<h3 id="as_estimator_spec"><code>as_estimator_spec</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py#L368-L395">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_estimator_spec()
</code></pre>

Creates an equivalent `EstimatorSpec` used by CPU train/eval.




