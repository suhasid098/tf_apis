description: Configuration for the "train" part for the train_and_evaluate call.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.estimator.TrainSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tf.estimator.TrainSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/training.py#L128-L198">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Configuration for the "train" part for the `train_and_evaluate` call.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.estimator.TrainSpec`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.estimator.TrainSpec(
    input_fn, max_steps=None, hooks=None, saving_listeners=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

`TrainSpec` determines the input data for the training, as well as the
duration. Optional hooks run at various stages of training.

#### Usage:



```
>>> train_spec = tf.estimator.TrainSpec(
...    input_fn=lambda: 1,
...    max_steps=100,
...    hooks=[_StopAtSecsHook(stop_after_secs=10)],
...    saving_listeners=[_NewCheckpointListenerForEvaluate(None, 20, None)])
>>> train_spec.saving_listeners[0]._eval_throttle_secs
20
>>> train_spec.hooks[0]._stop_after_secs
10
>>> train_spec.max_steps
100
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_fn`
</td>
<td>
A function that provides input data for training as minibatches.
See [Premade Estimators](
https://tensorflow.org/guide/premade_estimators#create_input_functions)
  for more information. The function should construct and return one of
the following:
  * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
    tuple (features, labels) with same constraints as below.
  * A tuple (features, labels): Where features is a `Tensor` or a
    dictionary of string feature name to `Tensor` and labels is a
    `Tensor` or a dictionary of string label name to `Tensor`.
</td>
</tr><tr>
<td>
`max_steps`
</td>
<td>
Int. Positive number of total steps for which to train model.
If `None`, train forever. The training `input_fn` is not expected to
generate `OutOfRangeError` or `StopIteration` exceptions. See the
`train_and_evaluate` stop condition section for details.
</td>
</tr><tr>
<td>
`hooks`
</td>
<td>
Iterable of `tf.train.SessionRunHook` objects to run on all workers
(including chief) during training.
</td>
</tr><tr>
<td>
`saving_listeners`
</td>
<td>
Iterable of <a href="../../tf/estimator/CheckpointSaverListener.md"><code>tf.estimator.CheckpointSaverListener</code></a>
objects to run on chief during training.
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
If any of the input arguments is invalid.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If any of the arguments is not of the expected type.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`input_fn`
</td>
<td>
A `namedtuple` alias for field number 0
</td>
</tr><tr>
<td>
`max_steps`
</td>
<td>
A `namedtuple` alias for field number 1
</td>
</tr><tr>
<td>
`hooks`
</td>
<td>
A `namedtuple` alias for field number 2
</td>
</tr><tr>
<td>
`saving_listeners`
</td>
<td>
A `namedtuple` alias for field number 3
</td>
</tr>
</table>



