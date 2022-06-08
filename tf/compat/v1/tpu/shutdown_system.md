description: Shuts down a running a distributed TPU system.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.tpu.shutdown_system" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.tpu.shutdown_system

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu.py">View source</a>



Shuts down a running a distributed TPU system.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.tpu.shutdown_system(
    job: Optional[Text] = None
) -> <a href="../../../../tf/Operation.md"><code>tf.Operation</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`job`
</td>
<td>
The job (the XXX in TensorFlow device specification /job:XXX) that
contains the TPU devices that will be shutdown. If job=None it is
assumed there is only one job in the TensorFlow flock, and an error will
be returned if this assumption does not hold.
</td>
</tr>
</table>

