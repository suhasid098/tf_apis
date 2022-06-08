description: Abstract base class used to build new callbacks.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.Callback" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="on_batch_begin"/>
<meta itemprop="property" content="on_batch_end"/>
<meta itemprop="property" content="on_epoch_begin"/>
<meta itemprop="property" content="on_epoch_end"/>
<meta itemprop="property" content="on_predict_batch_begin"/>
<meta itemprop="property" content="on_predict_batch_end"/>
<meta itemprop="property" content="on_predict_begin"/>
<meta itemprop="property" content="on_predict_end"/>
<meta itemprop="property" content="on_test_batch_begin"/>
<meta itemprop="property" content="on_test_batch_end"/>
<meta itemprop="property" content="on_test_begin"/>
<meta itemprop="property" content="on_test_end"/>
<meta itemprop="property" content="on_train_batch_begin"/>
<meta itemprop="property" content="on_train_batch_end"/>
<meta itemprop="property" content="on_train_begin"/>
<meta itemprop="property" content="on_train_end"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tf.keras.callbacks.Callback

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L575-L881">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Abstract base class used to build new callbacks.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.callbacks.Callback`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.Callback()
</code></pre>



<!-- Placeholder for "Used in" -->

Callbacks can be passed to keras methods such as `fit`, `evaluate`, and
`predict` in order to hook into the various stages of the model training and
inference lifecycle.

To create a custom callback, subclass <a href="../../../tf/keras/callbacks/Callback.md"><code>keras.callbacks.Callback</code></a> and override
the method associated with the stage of interest. See
https://www.tensorflow.org/guide/keras/custom_callback for more information.

#### Example:



```
>>> training_finished = False
>>> class MyCallback(tf.keras.callbacks.Callback):
...   def on_train_end(self, logs=None):
...     global training_finished
...     training_finished = True
>>> model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
>>> model.compile(loss='mean_squared_error')
>>> model.fit(tf.constant([[1.0]]), tf.constant([[1.0]]),
...           callbacks=[MyCallback()])
>>> assert training_finished == True
```

If you want to use `Callback` objects in a custom training loop:

1. You should pack all your callbacks into a single <a href="../../../tf/keras/callbacks/CallbackList.md"><code>callbacks.CallbackList</code></a>
   so they can all be called together.
2. You will need to manually call all the `on_*` methods at the appropriate
   locations in your loop. Like this:

   ```
   callbacks =  tf.keras.callbacks.CallbackList([...])
   callbacks.append(...)

   callbacks.on_train_begin(...)
   for epoch in range(EPOCHS):
     callbacks.on_epoch_begin(epoch)
     for i, data in dataset.enumerate():
       callbacks.on_train_batch_begin(i)
       batch_logs = model.train_step(data)
       callbacks.on_train_batch_end(i, batch_logs)
     epoch_logs = ...
     callbacks.on_epoch_end(epoch, epoch_logs)
   final_logs=...
   callbacks.on_train_end(final_logs)
   ```
The `logs` dictionary that callback methods
take as argument will contain keys for quantities relevant to
the current batch or epoch (see method-specific docstrings).



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`params`
</td>
<td>
Dict. Training parameters
(eg. verbosity, batch size, number of epochs...).
</td>
</tr><tr>
<td>
`model`
</td>
<td>
Instance of <a href="../../../tf/keras/Model.md"><code>keras.models.Model</code></a>.
Reference of the model being trained.
</td>
</tr>
</table>



## Methods

<h3 id="on_batch_begin"><code>on_batch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L650-L653">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_batch_begin(
    batch, logs=None
)
</code></pre>

A backwards compatibility alias for `on_train_batch_begin`.


<h3 id="on_batch_end"><code>on_batch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L655-L658">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_batch_end(
    batch, logs=None
)
</code></pre>

A backwards compatibility alias for `on_train_batch_end`.


<h3 id="on_epoch_begin"><code>on_epoch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L660-L671">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_begin(
    epoch, logs=None
)
</code></pre>

Called at the start of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`epoch`
</td>
<td>
Integer, index of epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_epoch_end"><code>on_epoch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L673-L687">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_end(
    epoch, logs=None
)
</code></pre>

Called at the end of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`epoch`
</td>
<td>
Integer, index of epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict, metric results for this training epoch, and for the
 validation epoch if validation is performed. Validation result keys
 are prefixed with `val_`. For training epoch, the values of the
`Model`'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
  0.7}`.
</td>
</tr>
</table>



<h3 id="on_predict_batch_begin"><code>on_predict_batch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L765-L780">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_predict_batch_begin(
    batch, logs=None
)
</code></pre>

Called at the beginning of a batch in `predict` methods.

Subclasses should override for any actions to run.

Note that if the `steps_per_execution` argument to `compile` in
<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> is set to `N`, this method will only be called every `N`
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch`
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_predict_batch_end"><code>on_predict_batch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L782-L796">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_predict_batch_end(
    batch, logs=None
)
</code></pre>

Called at the end of a batch in `predict` methods.

Subclasses should override for any actions to run.

Note that if the `steps_per_execution` argument to `compile` in
<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> is set to `N`, this method will only be called every `N`
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch`
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict. Aggregated metric results up until this batch.
</td>
</tr>
</table>



<h3 id="on_predict_begin"><code>on_predict_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L844-L853">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_predict_begin(
    logs=None
)
</code></pre>

Called at the beginning of prediction.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_predict_end"><code>on_predict_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L855-L864">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_predict_end(
    logs=None
)
</code></pre>

Called at the end of prediction.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_test_batch_begin"><code>on_test_batch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L726-L744">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_test_batch_begin(
    batch, logs=None
)
</code></pre>

Called at the beginning of a batch in `evaluate` methods.

Also called at the beginning of a validation batch in the `fit`
methods, if validation data is provided.

Subclasses should override for any actions to run.

Note that if the `steps_per_execution` argument to `compile` in
<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> is set to `N`, this method will only be called every `N`
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch`
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_test_batch_end"><code>on_test_batch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L746-L763">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_test_batch_end(
    batch, logs=None
)
</code></pre>

Called at the end of a batch in `evaluate` methods.

Also called at the end of a validation batch in the `fit`
methods, if validation data is provided.

Subclasses should override for any actions to run.

Note that if the `steps_per_execution` argument to `compile` in
<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> is set to `N`, this method will only be called every `N`
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch`
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict. Aggregated metric results up until this batch.
</td>
</tr>
</table>



<h3 id="on_test_begin"><code>on_test_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L821-L830">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_test_begin(
    logs=None
)
</code></pre>

Called at the beginning of evaluation or validation.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_test_end"><code>on_test_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L832-L842">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_test_end(
    logs=None
)
</code></pre>

Called at the end of evaluation or validation.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently the output of the last call to
`on_test_batch_end()` is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_train_batch_begin"><code>on_train_batch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L689-L706">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_train_batch_begin(
    batch, logs=None
)
</code></pre>

Called at the beginning of a training batch in `fit` methods.

Subclasses should override for any actions to run.

Note that if the `steps_per_execution` argument to `compile` in
<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> is set to `N`, this method will only be called every `N`
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch`
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_train_batch_end"><code>on_train_batch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L708-L724">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_train_batch_end(
    batch, logs=None
)
</code></pre>

Called at the end of a training batch in `fit` methods.

Subclasses should override for any actions to run.

Note that if the `steps_per_execution` argument to `compile` in
<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> is set to `N`, this method will only be called every `N`
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch`
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
`logs`
</td>
<td>
Dict. Aggregated metric results up until this batch.
</td>
</tr>
</table>



<h3 id="on_train_begin"><code>on_train_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L798-L807">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_train_begin(
    logs=None
)
</code></pre>

Called at the beginning of training.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_train_end"><code>on_train_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L809-L819">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_train_end(
    logs=None
)
</code></pre>

Called at the end of training.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently the output of the last call to `on_epoch_end()`
is passed to this argument for this method but that may change in
the future.
</td>
</tr>
</table>



<h3 id="set_model"><code>set_model</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L647-L648">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>




<h3 id="set_params"><code>set_params</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L644-L645">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>






