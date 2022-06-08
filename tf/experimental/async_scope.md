description: Context manager for grouping async operations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.async_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.async_scope

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/eager/context.py">View source</a>



Context manager for grouping async operations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.async_scope`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf_contextlib.contextmanager</code>
<code>tf.experimental.async_scope()
</code></pre>



<!-- Placeholder for "Used in" -->

Ops/function calls inside the scope can return before finishing the actual
execution. When exiting the async scope, a synchronization barrier will be
automatically added to ensure the completion of all async op and function
execution, potentially raising exceptions if async execution results in
an error state.

Users may write the following code to asynchronously invoke `train_step_fn`
and log the `loss` metric for every `num_steps` steps in a training loop.
`train_step_fn` internally consumes data using `iterator.get_next()`, and may
throw OutOfRangeError when running out of data. In the case:

```
try:
  with tf.experimental.async_scope():
    for _ in range(num_steps):
      # Step function updates the metric `loss` internally
      train_step_fn()
except tf.errors.OutOfRangeError:
  tf.experimental.async_clear_error()
logging.info('loss = %s', loss.numpy())
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
Context manager for grouping async operations.
</td>
</tr>

</table>

