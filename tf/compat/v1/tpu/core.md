description: Returns the device name for a core in a replicated TPU computation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.tpu.core" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.tpu.core

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_name_util.py">View source</a>



Returns the device name for a core in a replicated TPU computation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.tpu.core(
    num: int
) -> Text
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num`
</td>
<td>
the virtual core number within each replica to which operators should
be assigned.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A device name, suitable for passing to <a href="../../../../tf/device.md"><code>tf.device()</code></a>.
</td>
</tr>

</table>

