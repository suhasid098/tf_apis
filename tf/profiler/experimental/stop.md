description: Stops the current profiling session.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.profiler.experimental.stop" />
<meta itemprop="path" content="Stable" />
</div>

# tf.profiler.experimental.stop

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/profiler/profiler_v2.py">View source</a>



Stops the current profiling session.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.profiler.experimental.stop(
    save=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

The profiler session will be stopped and profile results can be saved.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`save`
</td>
<td>
An optional variable to save the results to TensorBoard. Default True.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`UnavailableError`
</td>
<td>
If there is no active profiling session.
</td>
</tr>
</table>

