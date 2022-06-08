description: Context manager for setting the executor of eager defined functions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.function_executor_type" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.function_executor_type

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/eager/context.py">View source</a>



Context manager for setting the executor of eager defined functions.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.function_executor_type`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf_contextlib.contextmanager</code>
<code>tf.experimental.function_executor_type(
    executor_type
)
</code></pre>



<!-- Placeholder for "Used in" -->

Eager defined functions are functions decorated by tf.contrib.eager.defun.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`executor_type`
</td>
<td>
a string for the name of the executor to be used to execute
functions defined by tf.contrib.eager.defun.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
Context manager for setting the executor of eager defined functions.
</td>
</tr>

</table>

