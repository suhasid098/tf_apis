description: Runs enclosed functions in the DTensor device scope.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.run_on" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.run_on

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Runs enclosed functions in the DTensor device scope.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@contextlib.contextmanager</code>
<code>tf.experimental.dtensor.run_on(
    layout_or_mesh: Union[<a href="../../../tf/experimental/dtensor/Layout.md"><code>tf.experimental.dtensor.Layout</code></a>, <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>]
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function returns a scope. All the ops and tf.functions in this scope will
run on the DTensor device using the mesh provided or attached to the layout.
This is useful for wrapping any tf.function that doesn't take a DTensor as
input but would like to produce DTensor as result. The scope will also make
sure all small constants be replicated as DTensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`layout_or_mesh`
</td>
<td>
A Layout or Mesh instance to extract a default mesh from.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
A context in which all ops and tf.functions will run on the DTensor device.
</td>
</tr>

</table>

