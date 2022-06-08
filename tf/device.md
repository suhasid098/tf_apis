description: Specifies the device for ops created/executed in this context.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.device" />
<meta itemprop="path" content="Stable" />
</div>

# tf.device

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Specifies the device for ops created/executed in this context.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.device(
    device_name
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function specifies the device to be used for ops created/executed in a
particular context. Nested contexts will inherit and also create/execute
their ops on the specified device. If a specific device is not required,
consider not using this function so that a device can be automatically
assigned.  In general the use of this function is optional. `device_name` can
be fully specified, as in "/job:worker/task:1/device:cpu:0", or partially
specified, containing only a subset of the "/"-separated fields. Any fields
which are specified will override device annotations from outer scopes.

#### For example:



```python
with tf.device('/job:foo'):
  # ops created here have devices with /job:foo
  with tf.device('/job:bar/task:0/device:gpu:2'):
    # ops created here have the fully specified device above
  with tf.device('/device:gpu:1'):
    # ops created here have the device '/job:foo/device:gpu:1'
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device_name`
</td>
<td>
The device name to use in the context.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A context manager that specifies the default device to use for newly
created ops.
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
If a function is passed in.
</td>
</tr>
</table>

