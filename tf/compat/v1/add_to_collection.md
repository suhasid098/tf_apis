description: Wrapper for Graph.add_to_collection() using the default graph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.add_to_collection" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.add_to_collection

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Wrapper for `Graph.add_to_collection()` using the default graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.add_to_collection(
    name, value
)
</code></pre>



<!-- Placeholder for "Used in" -->

See <a href="../../../tf/Graph.md#add_to_collection"><code>tf.Graph.add_to_collection</code></a>
for more details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
The key for the collection. For example, the `GraphKeys` class
contains many standard names for collections.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
The value to add to the collection.
</td>
</tr>
</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

Collections are only supported in eager when variables are created inside
an EagerVariableStore (e.g. as part of a layer or template).


 </devsite-expandable></section>

