description: Represents arguments to be added to a Session.run() call.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.estimator.SessionRunArgs" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tf.estimator.SessionRunArgs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/session_run_hook.py">View source</a>



Represents arguments to be added to a `Session.run()` call.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.estimator.SessionRunArgs`, `tf.compat.v1.train.SessionRunArgs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.estimator.SessionRunArgs(
    fetches, feed_dict=None, options=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fetches`
</td>
<td>
Exactly like the 'fetches' argument to Session.Run().
Can be a single tensor or op, a list of 'fetches' or a dictionary
of fetches.  For example:
  fetches = global_step_tensor
  fetches = [train_op, summary_op, global_step_tensor]
  fetches = {'step': global_step_tensor, 'summ': summary_op}
Note that this can recurse as expected:
  fetches = {'step': global_step_tensor,
             'ops': [train_op, check_nan_op]}
</td>
</tr><tr>
<td>
`feed_dict`
</td>
<td>
Exactly like the `feed_dict` argument to `Session.Run()`
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Exactly like the `options` argument to `Session.run()`, i.e., a
config_pb2.RunOptions proto.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`fetches`
</td>
<td>
A `namedtuple` alias for field number 0
</td>
</tr><tr>
<td>
`feed_dict`
</td>
<td>
A `namedtuple` alias for field number 1
</td>
</tr><tr>
<td>
`options`
</td>
<td>
A `namedtuple` alias for field number 2
</td>
</tr>
</table>



