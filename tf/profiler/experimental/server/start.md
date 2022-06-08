description: Start a profiler grpc server that listens to given port.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.profiler.experimental.server.start" />
<meta itemprop="path" content="Stable" />
</div>

# tf.profiler.experimental.server.start

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/profiler/profiler_v2.py">View source</a>



Start a profiler grpc server that listens to given port.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.profiler.experimental.server.start(
    port
)
</code></pre>



<!-- Placeholder for "Used in" -->

The profiler server will exit when the process finishes. The service is
defined in tensorflow/core/profiler/profiler_service.proto.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`port`
</td>
<td>
port profiler server listens to.
</td>
</tr>
</table>


Example usage: ```python tf.profiler.experimental.server.start(6009) # do
  your training here.