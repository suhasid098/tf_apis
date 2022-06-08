description: Container abstracting a list of callbacks.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.CallbackList" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="append"/>
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

# tf.keras.callbacks.CallbackList

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L182-L572">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Container abstracting a list of callbacks.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.callbacks.CallbackList`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.CallbackList(
    callbacks=None, add_history=False, add_progbar=False, model=None, **params
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`callbacks`
</td>
<td>
List of `Callback` instances.
</td>
</tr><tr>
<td>
`add_history`
</td>
<td>
Whether a `History` callback should be added, if one does not
already exist in the `callbacks` list.
</td>
</tr><tr>
<td>
`add_progbar`
</td>
<td>
Whether a `ProgbarLogger` callback should be added, if one
does not already exist in the `callbacks` list.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
The `Model` these callbacks are used with.
</td>
</tr><tr>
<td>
`**params`
</td>
<td>
If provided, parameters will be passed to each `Callback` via
<a href="../../../tf/keras/callbacks/Callback.md#set_params"><code>Callback.set_params</code></a>.
</td>
</tr>
</table>



## Methods

<h3 id="append"><code>append</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L274-L275">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>append(
    callback
)
</code></pre>




<h3 id="on_batch_begin"><code>on_batch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L381-L383">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_batch_begin(
    batch, logs=None
)
</code></pre>




<h3 id="on_batch_end"><code>on_batch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L385-L387">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_batch_end(
    batch, logs=None
)
</code></pre>




<h3 id="on_epoch_begin"><code>on_epoch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L389-L401">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_begin(
    epoch, logs=None
)
</code></pre>

Calls the `on_epoch_begin` methods of its callbacks.

This function should only be called during TRAIN mode.

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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L403-L416">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_end(
    epoch, logs=None
)
</code></pre>

Calls the `on_epoch_end` methods of its callbacks.

This function should only be called during TRAIN mode.

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
are prefixed with `val_`.
</td>
</tr>
</table>



<h3 id="on_predict_batch_begin"><code>on_predict_batch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L462-L472">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_predict_batch_begin(
    batch, logs=None
)
</code></pre>

Calls the `on_predict_batch_begin` methods of its callbacks.


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
Dict, contains the return value of `model.predict_step`,
it typically returns a dict with a key 'outputs' containing
the model's outputs.
</td>
</tr>
</table>



<h3 id="on_predict_batch_end"><code>on_predict_batch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L474-L482">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_predict_batch_end(
    batch, logs=None
)
</code></pre>

Calls the `on_predict_batch_end` methods of its callbacks.


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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L528-L537">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_predict_begin(
    logs=None
)
</code></pre>

Calls the 'on_predict_begin` methods of its callbacks.


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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L539-L548">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_predict_end(
    logs=None
)
</code></pre>

Calls the `on_predict_end` methods of its callbacks.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently, no data is passed via this argument
for this method, but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_test_batch_begin"><code>on_test_batch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L440-L450">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_test_batch_begin(
    batch, logs=None
)
</code></pre>

Calls the `on_test_batch_begin` methods of its callbacks.


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
Dict, contains the return value of `model.test_step`. Typically,
the values of the `Model`'s metrics are returned.  Example:
`{'loss': 0.2, 'accuracy': 0.7}`.
</td>
</tr>
</table>



<h3 id="on_test_batch_end"><code>on_test_batch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L452-L460">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_test_batch_end(
    batch, logs=None
)
</code></pre>

Calls the `on_test_batch_end` methods of its callbacks.


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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L506-L515">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_test_begin(
    logs=None
)
</code></pre>

Calls the `on_test_begin` methods of its callbacks.


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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L517-L526">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_test_end(
    logs=None
)
</code></pre>

Calls the `on_test_end` methods of its callbacks.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently, no data is passed via this argument
for this method, but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_train_batch_begin"><code>on_train_batch_begin</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L418-L428">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_train_batch_begin(
    batch, logs=None
)
</code></pre>

Calls the `on_train_batch_begin` methods of its callbacks.


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
Dict, contains the return value of `model.train_step`. Typically,
the values of the `Model`'s metrics are returned.  Example:
`{'loss': 0.2, 'accuracy': 0.7}`.
</td>
</tr>
</table>



<h3 id="on_train_batch_end"><code>on_train_batch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L430-L438">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_train_batch_end(
    batch, logs=None
)
</code></pre>

Calls the `on_train_batch_end` methods of its callbacks.


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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L484-L493">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_train_begin(
    logs=None
)
</code></pre>

Calls the `on_train_begin` methods of its callbacks.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently, no data is passed via this argument
for this method, but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_train_end"><code>on_train_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L495-L504">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_train_end(
    logs=None
)
</code></pre>

Calls the `on_train_end` methods of its callbacks.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logs`
</td>
<td>
Dict. Currently, no data is passed via this argument
for this method, but that may change in the future.
</td>
</tr>
</table>



<h3 id="set_model"><code>set_model</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L282-L287">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>




<h3 id="set_params"><code>set_params</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L277-L280">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/callbacks.py#L550-L551">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>






