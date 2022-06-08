description: Initialize the TPU devices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.initialize_tpu_system" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.initialize_tpu_system

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/tpu_util.py">View source</a>



Initialize the TPU devices.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.initialize_tpu_system(
    enable_coordination_service=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

This functions performs additional TPU related initialization after
calling <a href="../../../tf/experimental/dtensor/initialize_multi_client.md"><code>dtensor.initialize_multi_client</code></a> to initialize multi-client DTensor.
Refer to <a href="../../../tf/experimental/dtensor/initialize_multi_client.md"><code>dtensor.initialize_multi_client</code></a> for relevant environment
variables that controls the initialization of multi-client DTensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`enable_coordination_service`
</td>
<td>
If true, enable distributed coordination
service to make sure that workers know the devices on each other, a
prerequisite for data transfer through cross-worker rendezvous.
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
If running inside a tf.function.
</td>
</tr><tr>
<td>
`NotFoundError`
</td>
<td>
If no TPU devices found in eager mode.
</td>
</tr>
</table>

